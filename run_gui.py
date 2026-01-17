#!/usr/bin/env python
"""
Run the Gradio GUI.
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from gui.app import create_interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
