#!/usr/bin/env python3
"""
Simple HTTP server for viewing Flora visualizations.
Run this to serve the visualizations directory.
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path
import sys
import os

PORT = 8000
VIZ_DIR = "visualizations"
THUMBNAIL_DIR = "thumbnails_cache"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve from the root of the project, so both visualizations and thumbnails are accessible
        super().__init__(*args, directory=".", **kwargs)

    def log_message(self, format, *args):
        # Pretty print logs
        print(f"[{self.log_date_time_string()}] {format % args}")

    def do_GET(self):
        # If the request is for the root, serve the index.html from the visualizations directory
        if self.path == '/':
            self.path = f'/{VIZ_DIR}/index.html'
        
        # If the request is for a file in the visualizations directory, rewrite the path
        elif self.path.startswith(f'/{VIZ_DIR}/'):
            pass # Path is already correct

        # If the request is for a thumbnail, rewrite the path
        elif self.path.startswith('/thumbnails/'):
            self.path = self.path.replace('/thumbnails/', f'/{THUMBNAIL_DIR}/')

        # For other HTML files, assume they are in the visualizations directory
        elif self.path.endswith('.html'):
            if not self.path.startswith('/'):
                self.path = '/' + self.path
            if not self.path.startswith(f'/{VIZ_DIR}/'):
                self.path = f'/{VIZ_DIR}{self.path}'
        
        super().do_GET()

def main():
    # Check if visualizations directory exists
    viz_dir = Path(VIZ_DIR)
    if not viz_dir.exists():
        print(f"❌ Error: '{VIZ_DIR}/' directory not found")
        print("   Run 'python visualize_colors.py' first to generate visualizations")
        sys.exit(1)

    # Check for cluster data
    cluster_dir = viz_dir / "cluster_data"
    has_clusters = (cluster_dir.exists() and
                    len(list(cluster_dir.glob("*.json"))) > 0)

    # Allow reusing the address to avoid "Address already in use" errors
    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("=" * 60)
            print("🌿 Flora Visualizations Server")
            print("=" * 60)
            print(f"\n📁 Serving from: {Path('.').absolute()}")
            print(f"🌐 Server URL: http://localhost:{PORT}/")
            
            cluster_status = '✓ Available' if has_clusters else '✗ Not found'
            print(f"🧬 Cluster data: {cluster_status}")
            
            print("\n📊 Available visualizations:")
            print(f"   • Main dashboard:        http://localhost:{PORT}/")
            print(f"   • 3D Color Space:        http://localhost:{PORT}/color_space_3d.html")
            print(f"   • Plant Color Grid:      http://localhost:{PORT}/plant_color_grid.html")
            if has_clusters:
                print(f"   • Plants by Cluster:     http://localhost:{PORT}/plant_colors_by_cluster.html")
            
            print("\n" + "=" * 60)
            print("Press Ctrl+C to stop the server")
            print("=" * 60 + "\n")

            # Open browser
            url = f"http://localhost:{PORT}/"
            print(f"🚀 Opening browser to {url}\n")
            webbrowser.open(url)

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        sys.exit(0)
    except OSError as e:
        print(f"❌ An unexpected error occurred: {e}")
        print(f"   It's possible another process is using port {PORT}.")
        print(f"   You can find the process ID (PID) with: lsof -i :{PORT}")
        sys.exit(1)

if __name__ == "__main__":
    main()