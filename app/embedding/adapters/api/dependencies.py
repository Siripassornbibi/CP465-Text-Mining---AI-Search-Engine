"""
adapters/api/dependencies.py — FastAPI dependency injection.
"""
from __future__ import annotations
from typing import Annotated
 
from fastapi import Depends, Request
 
from embedding.application.search import SearchUseCase
from embedding.application.expand_query import ExpandQueryUseCase
 
 
def get_search_use_case(request: Request) -> SearchUseCase:
    return request.app.state.container.search_use_case
 
 
def get_expand_use_case(request: Request) -> ExpandQueryUseCase:
    return request.app.state.container.expand_query_use_case
 
 
SearchUseCaseDep  = Annotated[SearchUseCase,  Depends(get_search_use_case)]
ExpandUseCaseDep  = Annotated[ExpandQueryUseCase, Depends(get_expand_use_case)]
 