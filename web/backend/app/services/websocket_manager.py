# Copyright (c) US Inc. All rights reserved.
"""WebSocket manager for real-time updates"""

import asyncio
import json
from typing import Dict, List, Set
from fastapi import WebSocket


class WebSocketManager:
    """Manages WebSocket connections for real-time job updates"""
    
    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        """Accept and register a WebSocket connection"""
        await websocket.accept()
        async with self._lock:
            if job_id not in self._connections:
                self._connections[job_id] = set()
            self._connections[job_id].add(websocket)
    
    async def disconnect(self, websocket: WebSocket, job_id: str) -> None:
        """Remove a WebSocket connection"""
        async with self._lock:
            if job_id in self._connections:
                self._connections[job_id].discard(websocket)
                if not self._connections[job_id]:
                    del self._connections[job_id]
    
    async def broadcast(self, job_id: str, message: dict) -> None:
        """Broadcast message to all connections for a job"""
        async with self._lock:
            if job_id not in self._connections:
                return
            
            dead_connections = set()
            for websocket in self._connections[job_id]:
                try:
                    await websocket.send_json(message)
                except Exception:
                    dead_connections.add(websocket)
            
            # Clean up dead connections
            for ws in dead_connections:
                self._connections[job_id].discard(ws)
    
    async def send_log(self, job_id: str, log_line: str, log_type: str = "info") -> None:
        """Send a log line to all connections"""
        await self.broadcast(job_id, {
            "type": "log",
            "log_type": log_type,
            "message": log_line,
        })
    
    async def send_progress(self, job_id: str, step: int, total: int, 
                           loss: float = None, lr: float = None) -> None:
        """Send training progress update"""
        await self.broadcast(job_id, {
            "type": "progress",
            "step": step,
            "total_steps": total,
            "loss": loss,
            "learning_rate": lr,
            "progress_percent": round((step / total) * 100, 2) if total > 0 else 0,
        })
    
    async def send_status(self, job_id: str, status: str, error: str = None) -> None:
        """Send status update"""
        await self.broadcast(job_id, {
            "type": "status",
            "status": status,
            "error": error,
        })
    
    def get_connection_count(self, job_id: str) -> int:
        """Get number of connections for a job"""
        return len(self._connections.get(job_id, set()))


# Global instance
ws_manager = WebSocketManager()
