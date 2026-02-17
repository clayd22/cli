"""
render_artifact.py

Tool for rendering interactive visualizations served on a local port.

ARCHITECTURE (Anti-Hallucination):
- LLM writes the visualization CODE (HTML/CSS/JS)
- System executes SQL queries to get REAL DATA
- System injects real data as JSON into the HTML
- System serves on local port and opens browser
- LLM cannot fake the data - it only defines how to display it

The code expects a DATA object to be available, which the system provides
from actual database query results.
"""

import json
import socket
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from ...sandbox import SQLExecutor


# Output directory for rendered artifacts
ARTIFACTS_DIR = Path(__file__).parent.parent.parent.parent / ".astroagent" / "artifacts"

# Track running servers
_running_servers = []


RENDER_ARTIFACT_TOOL = {
    "type": "function",
    "function": {
        "name": "render_artifact",
        "description": """Render an interactive visualization served on a local port.

HOW IT WORKS:
1. You provide SQL queries to fetch data
2. You write complete HTML/CSS/JavaScript code for the visualization
3. System executes SQL, injects real data as window.DATA, serves on local port, opens browser

YOUR CODE HAS ACCESS TO:
- window.DATA: Object containing query results (each key = query name, value = array of row objects)
- Any JS libraries via CDN (Three.js, D3, Chart.js, Plotly, etc.)

QUALITY REQUIREMENTS:
1. Write COMPLETE, valid HTML5 documents (<!DOCTYPE html>, <html>, <head>, <body>)
2. Include proper <meta charset="UTF-8"> and viewport meta tag
3. Add error handling: check if window.DATA exists before using
4. Make visualizations responsive (use percentages or viewport units)
5. Add loading states and error messages for better UX
6. Use modern JS (const/let, arrow functions, template literals)
7. Add meaningful titles and labels to charts
8. Use readable color schemes with good contrast
9. Test data access: console.log(window.DATA) to debug

TEMPLATE STRUCTURE:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Visualization Title</title>
  <script src="CDN_LIBRARY_URL"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; }
    #container { width: 100vw; height: 100vh; }
    .error { color: red; padding: 20px; }
  </style>
</head>
<body>
  <div id="container"></div>
  <script>
    // Check data exists
    if (!window.DATA || !window.DATA.queryname) {
      document.getElementById('container').innerHTML = '<p class="error">Data not loaded</p>';
    } else {
      const data = window.DATA.queryname;
      // Build visualization...
    }
  </script>
</body>
</html>
```

FOR 3D VISUALIZATIONS (Three.js):
- Use OrbitControls for camera rotation
- Add proper lighting (ambient + directional)
- Handle window resize events
- Add animation loop with requestAnimationFrame

The DATA is injected by the system from real SQL results - you cannot hardcode data values.""",
        "parameters": {
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "object",
                    "description": "Map of names to SQL queries. Results become window.DATA.{name} as arrays of objects.",
                    "additionalProperties": {
                        "type": "string"
                    }
                },
                "code": {
                    "type": "string",
                    "description": "Complete HTML document with embedded CSS/JS. Access data via window.DATA.{input_name}. Include any libraries via CDN."
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename (e.g., 'chart.html', 'visualization.html')"
                },
                "explanation": {
                    "type": "string",
                    "description": "What this visualization shows"
                }
            },
            "required": ["inputs", "code", "filename", "explanation"]
        }
    }
}


@dataclass
class RenderArtifactOutput:
    """Container for render artifact output."""
    success: bool
    file_path: Path = None
    url: str = None
    explanation: str = None
    error: str = None


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class QuietHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler that doesn't log to console."""
    def log_message(self, format, *args):
        pass  # Suppress logging


def start_server(directory: Path, port: int) -> HTTPServer:
    """Start an HTTP server serving the given directory."""
    handler = partial(QuietHTTPHandler, directory=str(directory))
    server = HTTPServer(('localhost', port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _running_servers.append(server)
    return server


def render_artifact(
    inputs: dict[str, str],
    code: str,
    filename: str,
    explanation: str
) -> RenderArtifactOutput:
    """
    Execute SQL queries, inject data into HTML code, and open in browser.

    This maintains the anti-hallucination guarantee:
    - LLM writes the visualization structure (HTML/JS)
    - DATA comes from actual SQL execution
    - LLM cannot fake data values

    Args:
        inputs: Map of names to SQL queries
        code: HTML/CSS/JS code expecting window.DATA
        filename: Output filename
        explanation: What this shows

    Returns:
        RenderArtifactOutput with file path or error
    """
    sql_executor = SQLExecutor()

    # Ensure artifacts directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / filename

    # Execute SQL queries and convert to JSON-serializable format
    data = {}
    for name, sql in inputs.items():
        df, error = sql_executor.execute(sql)
        if error:
            return RenderArtifactOutput(
                success=False,
                error=f"SQL error for '{name}': {error}",
                explanation=explanation
            )
        # Convert DataFrame to list of dicts (JSON-serializable)
        data[name] = df.to_dict(orient='records')

    try:
        # Inject data into HTML
        # Find </head> or <body> or start of <script> to inject data script
        data_script = f"<script>window.DATA = {json.dumps(data, default=str)};</script>\n"

        if "</head>" in code:
            html = code.replace("</head>", f"{data_script}</head>")
        elif "<body>" in code:
            html = code.replace("<body>", f"<body>\n{data_script}")
        elif "<script>" in code:
            html = code.replace("<script>", f"{data_script}<script>", 1)
        else:
            # Prepend to code
            html = data_script + code

        # Write to file
        output_path.write_text(html)

        # Start local server
        port = find_free_port()
        start_server(ARTIFACTS_DIR, port)
        url = f"http://localhost:{port}/{filename}"

        # Open in browser
        webbrowser.open(url)

        return RenderArtifactOutput(
            success=True,
            file_path=output_path,
            url=url,
            explanation=explanation
        )

    except Exception as e:
        import traceback
        return RenderArtifactOutput(
            success=False,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            explanation=explanation
        )
