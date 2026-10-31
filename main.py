"""
VisionGuard Application Entry Point
Allows running directly with:
    uvicorn main:app --reload
or:
    python main.py
"""

from visionguard.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

