# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-06-26

### Added
- **Google ADK Integration**: Migrated the entire Garmin Fitness Assistant from CrewAI to Google's Agent Development Kit (ADK).
- **ADK Agent Package (`garmin_agent/`)**:
  - `agent.py`: Defined the `root_agent` using `google.adk.agents.Agent` and configured system instructions and Gemini model.
  - `tools.py`: Exposes 13 Garmin Connect API functions as standard Python functions decorated as ADK tools.
  - `__init__.py`: Imports the agent to make it discoverable by the ADK framework.
- **Real-Time Progress Indicators**: Added explicit progress logging and emojis to all tools in `garmin_agent/tools.py` to provide real-time feedback in the terminal during multi-step agent executions.
- **Interactive Web UI**: Added support for ADK's built-in web chat interface via `pixi run web-chat`.
- **Comprehensive Agent Guide (`AGENTS.md`)**: Created a high-signal guide documenting repository-specific quirks, package spelling, environment management, and developer commands to help future agents and developers ramp up instantly.

### Changed
- **Environment Management**: Migrated the `pixi` configuration to the root directory for cleaner environment management.
- **Pixi Tasks**: Updated the pixi tasks in `pyproject.toml` to use `adk run garmin_agent` and `adk web garmin_agent`.
- **Root Documentation**: Rebranded `README.md` to fully reflect the new Google ADK architecture, features, and commands.

### Fixed
- **Environment Loading**: Added `load_dotenv()` to all Python entry points to ensure credentials are automatically loaded from the local `.env` file.
- **Robust Tool Execution**: Wrapped all 13 Garmin Connect tools in robust `try...except` blocks to handle API limits or invalid filters gracefully without crashing.

---

# Release Notes - v1.0.0

## 🏋️‍♂️ Google ADK Garmin Fitness Assistant - Production Release

We are thrilled to announce the official **v1.0.0** release of the **Garmin Fitness Assistant**, fully migrated to the **Google Agent Development Kit (ADK)**!

This release marks a major milestone, transitioning the project from CrewAI to Google's native, code-first framework for building, orchestrating, and deploying Gemini-powered agents.

### 🌟 Key Highlights

#### 1. Google ADK Architecture
The assistant is now built on Google's modern Agent Development Kit. This provides a highly modular, code-first approach to agent development, allowing for seamless tool execution and orchestration using Gemini models.

#### 2. Robust Tool Execution & Error Handling
All 13 Garmin Connect tools (including activity tracking, sleep analysis, heart rate monitoring, stress levels, and body battery metrics) are now wrapped in robust `try...except` blocks. If the model attempts to use an invalid filter (e.g., `"mountain biking"` instead of `"cycling"`), the tool handles the error gracefully and provides helpful suggestions, allowing the model to automatically self-correct and retry without crashing.

#### 3. Real-Time Progress Feedback
To solve the "silent terminal" UX issue, we've added explicit progress logging to all tools. You will now see real-time updates (e.g., `🔑 [Tool] Authenticating...`, `🏃‍♂️ [Tool] Fetching recent activities...`) as the agent executes multi-step processes behind the scenes.

#### 4. Interactive Web UI
In addition to the interactive CLI chat, you can now launch a beautiful, built-in web chat interface to interact with your fitness assistant using:
```bash
pixi run web-chat
```

#### 5. Clean Environment Management
The project is fully managed by `pixi` with tasks configured in the root `pyproject.toml`. Credentials are kept secure and local-only in the `.env` file, which is automatically loaded on startup.

---

### 🚀 Getting Started

To run the new ADK assistant, ensure your `.env` file is configured in the root directory, and run:

```bash
# Run CLI Chat
pixi run run-garmin-chat

# Run Web UI Chat (opens on http://localhost:8000)
pixi run web-chat
```
