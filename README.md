# Satellite Infrastructure Scanner

This project provides a FastAPI web application that analyzes satellite imagery with
state-of-the-art vision-language models. Users can upload satellite scenes, request a
custom analysis prompt (defaulting to "Describe all unusual objects in this image"), and the
system stores detected objects and metadata in a SQLite database for later review.

Key features:

- **Image captioning** using BLIP to summarize the context of the uploaded scene.
- **Promptable visual question answering** to respond to analyst questions such as the default
  "Describe all unusual objects in this image".
- **General object detection** powered by a DETR model capable of identifying a broad range of
  infrastructure and equipment beyond a fixed list of examples.
- **Persistent storage** of every analysis run, including references to uploaded imagery.

## Getting started

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -U pip
   pip install -e .
   ```

   The first run will download the required Hugging Face models (BLIP for captioning and VILT for
   visual question answering). Expect this to take a few minutes.

3. Launch the development server:

   ```bash
   uvicorn app.main:app --reload
   ```

4. Open <http://localhost:8000> in your browser, upload a satellite image, optionally adjust the
   analysis prompt, and submit the form. The results table will grow with each analysis, allowing
   you to revisit previous detections.

Uploaded imagery is stored under `data/uploads`, and analysis metadata is tracked in the
`data/satellite_scans.db` SQLite database.

## Project structure

```
app/
  main.py                # FastAPI application and routes
  database.py            # Database configuration and helpers
  models.py              # SQLModel ORM models
  services/
    analyzer.py          # Vision-language analysis service
  templates/
    index.html           # Jinja2 template for the UI
  static/
    styles.css           # Styling for the interface
```

## Notes

- The object detector returns bounding boxes and confidence scores for any recognized objects,
  enabling analysts to scan for a wide range of infrastructure without curating prompts.
- The application is designed for CPU inference. Larger scenes may take several seconds to
  process because the models run sequentially.
- For production deployments consider adding background task processing, authentication, and
  storage in a more scalable database.
