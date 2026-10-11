from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from memmamba.config import foreground_classes
from memmamba.memory.storage import MultimodalMedicalKnowledgeMemory

@dataclass
class ClasswiseRetrievalResult:
    visual_key:Any; interior:Any; boundary:Any; context:Any; semantic:Any; scores:Any; indices:Any; class_ids:Any; valid_mask:Any

def classwise_topk(projected_queries,memory:MultimodalMedicalKnowledgeMemory,top_k:int,active_mask=None):
    import torch
    import torch.nn.functional as functional
    memory.validate()
    if projected_queries.ndim!=3 or top_k<=0 or projected_queries.shape[-1]!=memory.visual_dim: raise ValueError("queries/top_k incompatible with Memory")
    organs=foreground_classes(memory.classes); batch,class_count,dim=projected_queries.shape
    if class_count!=len(organs): raise ValueError("query foreground count differs")
    if active_mask is None: active_mask=torch.ones(batch,class_count,dtype=torch.bool,device=projected_queries.device)
    if active_mask.shape!=(batch,class_count) or active_mask.dtype!=torch.bool or active_mask.device!=projected_queries.device: raise ValueError("active_mask must be bool [B,C-1] on query device")
    k=int(top_k); device=projected_queries.device; bank=memory.to(device); queries=functional.normalize(projected_queries,dim=-1)
    def zeros(d): return torch.zeros(batch,class_count,k,d,device=device)
    out={"visual_key":zeros(dim),"interior":zeros(dim),"boundary":zeros(dim),"context":zeros(dim),"semantic":zeros(bank.semantic_dim)}
    scores=torch.full((batch,class_count,k),float("-inf"),device=device); indices=torch.full((batch,class_count,k),-1,dtype=torch.long,device=device)
    ids=torch.zeros((batch,class_count,k),dtype=torch.long,device=device); valid=torch.zeros((batch,class_count,k),dtype=torch.bool,device=device)
    for ci,info in enumerate(organs):
        active_rows=torch.nonzero(active_mask[:,ci],as_tuple=False).flatten()
        if active_rows.numel()==0: continue
        candidates=bank.indices_for_class(info.id); keep=min(k,len(candidates))
        values,local=torch.topk(queries[active_rows,ci]@functional.normalize(bank.visual_key[candidates],dim=-1).T,keep,dim=-1); selected=candidates[local]
        scores[active_rows,ci,:keep]=values; indices[active_rows,ci,:keep]=selected; valid[active_rows,ci,:keep]=True; ids[active_rows,ci,:keep]=info.id
        for name in out: out[name][active_rows,ci,:keep]=getattr(bank,name)[selected]
    return ClasswiseRetrievalResult(out["visual_key"],out["interior"],out["boundary"],out["context"],out["semantic"],scores,indices,ids,valid)
