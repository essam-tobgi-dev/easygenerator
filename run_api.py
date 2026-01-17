#!/usr/bin/env python
"""
Run the FastAPI server.
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
