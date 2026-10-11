from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as functional

from memmamba.memory.alignment_teachers import (
    STRICT_SOURCE,
    AlignmentTeacherValue,
)

class DiceCELoss(nn.Module):
    def __init__(self,num_classes:int,dice_weight:float=1.,ce_weight:float=1.,smooth:float=1e-5):
        super().__init__(); self.num_classes=num_classes; self.dice_weight=dice_weight; self.ce_weight=ce_weight; self.smooth=smooth
        if num_classes<2 or min(dice_weight,ce_weight)<0: raise ValueError("invalid DiceCE configuration")
    def forward(self,logits,target):
        if logits.ndim!=4 or target.shape!=(logits.shape[0],*logits.shape[-2:]) or logits.shape[1]!=self.num_classes or logits.device!=target.device: raise ValueError("logits/target shape or device mismatch")
        ce=functional.cross_entropy(logits,target); probs=functional.softmax(logits,1); onehot=functional.one_hot(target,num_classes=self.num_classes).permute(0,3,1,2).float()
        inter=(probs[:,1:]*onehot[:,1:]).sum((0,2,3)); denom=(probs[:,1:]+onehot[:,1:]).sum((0,2,3)); dice=1-((2*inter+self.smooth)/(denom+self.smooth)).mean()
        return self.dice_weight*dice+self.ce_weight*ce

def presence_targets(label,num_classes):
    if label.ndim!=3 or num_classes<2: raise ValueError("label must be [B,H,W]")
    return torch.stack([(label==cid).reshape(label.shape[0],-1).any(dim=1) for cid in range(1,num_classes)],1).float()

@dataclass
class AlignmentTeacherBatch:
    teacher: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor
    strict_count: int
    missing_count: int


def query_alignment_loss(query,teacher,valid,confidence):
    if not all(torch.is_tensor(value) for value in (query, teacher, valid, confidence)):
        raise ValueError("alignment inputs must be tensors")
    if query.ndim != 3 or teacher.ndim != 3 or query.shape != teacher.shape:
        raise ValueError("query and teacher must have the same [B,C,D] shape")
    if valid.shape != query.shape[:2] or confidence.shape != query.shape[:2]:
        raise ValueError("valid and confidence must have shape [B,C]")
    if not query.is_floating_point() or not teacher.is_floating_point():
        raise ValueError("query and teacher must use floating-point dtypes")
    if query.dtype != teacher.dtype:
        raise ValueError("query and teacher must use the same dtype")
    if valid.dtype != torch.bool:
        raise ValueError("valid must use bool dtype")
    if not confidence.is_floating_point():
        raise ValueError("confidence must use a floating-point dtype")
    if any(value.device != query.device for value in (teacher, valid, confidence)):
        raise ValueError("alignment tensors must be on the same device")
    if not torch.isfinite(query).all() or not torch.isfinite(teacher).all():
        raise ValueError("query and teacher must contain only finite values")
    if not torch.isfinite(confidence).all():
        raise ValueError("confidence must contain only finite values")
    if torch.any(confidence[valid] <= 0.0) or torch.any(confidence[valid] > 1.0):
        raise ValueError("valid confidence values must lie in (0,1]")
    if torch.any(confidence[~valid] != 0.0):
        raise ValueError("invalid confidence values must equal zero")
    if not bool(valid.any()):
        return query.sum()*0.

    weights = confidence * valid.to(dtype=confidence.dtype)
    errors = 1.0 - functional.cosine_similarity(query, teacher, dim=-1)
    denominators = weights.sum(dim=0)
    active = denominators > 0.0
    numerators = (errors * weights).sum(dim=0)
    return (numerators[active] / denominators[active]).mean()


def alignment_teacher_batch(case_ids,slice_ids,label,lookup,class_count,feature_dim,device):
    if not torch.is_tensor(label) or label.ndim != 3:
        raise ValueError("label must be a [B,H,W] tensor")
    if len(case_ids) != label.shape[0] or len(slice_ids) != label.shape[0]:
        raise ValueError("case_ids and slice_ids must match the label batch size")
    if type(class_count) is not int or class_count <= 0:
        raise ValueError("class_count must be a positive integer")
    if type(feature_dim) is not int or feature_dim <= 0:
        raise ValueError("feature_dim must be a positive integer")
    teacher=torch.zeros(len(case_ids),class_count,feature_dim,device=device)
    valid=torch.zeros(len(case_ids),class_count,dtype=torch.bool,device=device)
    confidence=torch.zeros(len(case_ids),class_count,dtype=teacher.dtype,device=device)
    strict_count=missing_count=0
    for b,(case_id,slice_id) in enumerate(zip(case_ids,slice_ids)):
        if torch.is_tensor(slice_id):
            if slice_id.numel() != 1:
                raise ValueError("slice_id tensor must be scalar")
            normalized_slice_id = int(slice_id.item())
        else:
            normalized_slice_id = int(slice_id)
        for cid in range(1,class_count+1):
            if not bool((label[b]==cid).any()): continue
            value=lookup.get((str(case_id),normalized_slice_id,cid))
            if value is None:
                missing_count+=1
                continue
            if not isinstance(value, AlignmentTeacherValue):
                raise ValueError("alignment teacher lookup values must be AlignmentTeacherValue")
            if value.source != STRICT_SOURCE:
                raise ValueError("invalid alignment teacher source")
            if not torch.is_tensor(value.feature) or value.feature.ndim!=1 or value.feature.shape[0]!=feature_dim or not torch.isfinite(value.feature).all():
                raise ValueError("invalid alignment teacher vector")
            if isinstance(value.confidence, bool) or not isinstance(value.confidence, (int, float)):
                raise ValueError("alignment teacher confidence must be numeric")
            teacher_confidence=float(value.confidence)
            if not math.isfinite(teacher_confidence) or not 0.0 < teacher_confidence <= 1.0:
                raise ValueError("alignment teacher confidence must lie in (0,1]")
            teacher[b,cid-1]=value.feature.to(device=device,dtype=teacher.dtype)
            valid[b,cid-1]=True
            confidence[b,cid-1]=teacher_confidence
            strict_count+=1
    return AlignmentTeacherBatch(
        teacher=teacher,
        valid=valid,
        confidence=confidence,
        strict_count=strict_count,
        missing_count=missing_count,
    )

@dataclass
class TrainingLoss:
    total:torch.Tensor; final:torch.Tensor; coarse:torch.Tensor; presence:torch.Tensor; alignment:torch.Tensor

def compute_training_loss(output,label,criterion,lambda_aux,lambda_presence,lambda_align,teacher=None,teacher_valid=None,teacher_confidence=None):
    if min(lambda_aux,lambda_presence,lambda_align)<0: raise ValueError("loss lambdas must be non-negative")
    final=criterion(output.final_logits,label); coarse=criterion(output.coarse_logits,label); targets=presence_targets(label,output.final_logits.shape[1]).to(output.presence_logits.device)
    if output.presence_logits.shape!=targets.shape: raise ValueError("presence logits shape mismatch")
    presence=functional.binary_cross_entropy_with_logits(output.presence_logits,targets)
    teacher_bundle=(teacher,teacher_valid,teacher_confidence)
    if any(value is not None for value in teacher_bundle) and not all(value is not None for value in teacher_bundle):
        raise ValueError("teacher, teacher_valid, and teacher_confidence must be provided together")
    alignment=final*0. if output.projected_query is None or teacher is None else query_alignment_loss(output.projected_query,teacher,teacher_valid,teacher_confidence)
    return TrainingLoss(final+lambda_aux*coarse+lambda_presence*presence+lambda_align*alignment,final,coarse,presence,alignment)
