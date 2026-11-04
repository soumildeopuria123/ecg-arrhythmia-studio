# Contributing

Thanks for taking a look at ECG Arrhythmia Studio. This repository is a
monorepo containing three packages:

| Package | Purpose |
| --- | --- |
| `packages/ml-engine` | Signal preprocessing, STFT conversion, model inference and Grad-CAM |
| `packages/backend` | FastAPI service that accepts signals and runs predictions |
| `packages/frontend` | Next.js dashboard for uploading signals and reviewing results |

## Local setup

1. Create a virtual environment for the Python packages.
2. Install the backend and ML engine requirements.
3. Install the frontend dependencies.
4. Start the services with `docker compose up --build` or run them directly.

## Making changes

- Keep pull requests focused on a single change.
- Add or update tests when behaviour changes.
- Match the existing code style of the package you are touching.
- Run the relevant test suite before opening a pull request.

## Reporting issues

Use the issue template and include the commit you tested against, the command
you ran, and the full error output where possible.
