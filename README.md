# Satellite Infrastructure Scanner

This project provides a FastAPI web application that analyzes satellite imagery with
state-of-the-art vision-language models. Users can upload satellite scenes, request a
custom analysis prompt (defaulting to "Describe all unusual objects in this image"), and the
system stores detected objects and metadata in a SQLite database for later review.

Key features:

- **Image captioning** using BLIP-2 to summarize the context of the uploaded scene.
- **Promptable visual question answering** to respond to analyst questions such as the default
  "Describe all unusual objects in this image".
- **General object detection** powered by a DETR model capable of identifying a broad range of
  infrastructure and equipment beyond a fixed list of examples.
- **Persistent storage** of every analysis run, including references to uploaded imagery.
- **Automatic area scanning** by fetching imagery for a bounding box from MapTiler's global satellite
  basemap (high-resolution by default) with optional NASA GIBS mosaics and USGS NAIP aerial
  photography, analyzing each tile without preparing archives.

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

   The first run will download the required Hugging Face models (BLIP-2 for captioning and VILT for
   visual question answering). Expect this to take a few minutes.

3. Configure your MapTiler API key (required for the default imagery provider):

   ```bash
   export MAPTILER_API_KEY="your_maptiler_api_key"
   # Optional: set a Referer header if your MapTiler key is domain-restricted
   # (for example when only localhost requests are allowed).
   export MAPTILER_REFERER="http://localhost:8000"
   ```

   You can create and manage keys from your MapTiler Cloud account. The downloader now uses the
   Raster Tiles API and composites the required MapTiler attribution overlay locally, so the key
   must include Raster (or Rendered Maps) access. Keys are read from the environment at runtime, so
   shell configuration such as `.env` files also works.

4. Launch the development server:

   ```bash
   uvicorn app.main:app --reload
   ```

5. Open <http://localhost:8000> in your browser. You can either upload a single satellite image or
   use the automatic area scan form to request imagery for a latitude/longitude bounding box. By
   default the app retrieves crisp MapTiler satellite imagery centered on Vienna, Austria, but you
   can switch to NASA's Global Imagery Browse Services (GIBS) or the USGS NAIP aerial mosaics when
   you need alternate coverage. Each scan is limited to 50 imagery requests to prevent accidental
   overload of the services, and every tile is requested at a minimum of 256×256 pixels so the
   vision models have enough detail to work with even for small bounding boxes. When the NASA GIBS
   VIIRS true-color layer is too blocky for object recognition, the downloader automatically retries
   with the higher resolution Landsat WELD mosaic to deliver ~30 m/pixel imagery.

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
