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
- **Automatic area scanning** by fetching NASA GIBS imagery tiles for a bounding box and analyzing
  them without preparing archives.

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

4. Open <http://localhost:8000> in your browser. You can either upload a single satellite image or
   use the automatic area scan form to request imagery for a latitude/longitude bounding box. The
   app downloads tiles from NASA's Global Imagery Browse Services (GIBS), analyzes each tile, and
   stores the results so you can revisit previous detections. Each scan is limited to 50 GIBS
   requests to prevent accidental overload of the service.

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
