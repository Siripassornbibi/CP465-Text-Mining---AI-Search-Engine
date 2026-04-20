"""
adapters/api/dependencies.py — FastAPI dependency injection.
Pulls singletons from the container instead of creating new instances per request.
"""
from __future__ import annotations
from typing import Annotated

from fastapi import Depends, Request

from crawlerapp.application.search import SearchUseCase


def get_search_use_case(request: Request) -> SearchUseCase:
    return request.app.state.container.search_use_case


SearchUseCaseDep = Annotated[SearchUseCase, Depends(get_search_use_case)]
