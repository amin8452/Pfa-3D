"""
Script to set up the GitHub repository for Hunyuan3D-Glasses
"""

import os
import subprocess
import argparse


def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def setup_github(username, repo_name="Hunyuan3D-Glasses", remote_url=None):
    """
    Set up the GitHub repository
    
    Args:
        username: GitHub username
        repo_name: Repository name
        remote_url: Remote URL (if already created)
    """
    # Initialize git repository if not already initialized
    if not os.path.exists(".git"):
        run_command("git init")
    
    # Add all files
    run_command("git add .")
    
    # Commit
    run_command('git commit -m "Initial commit: Hunyuan3D-Glasses project setup"')
    
    # Set up remote
    if remote_url is None:
        remote_url = f"https://github.com/{username}/{repo_name}.git"
    
    # Check if remote already exists
    result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
    if "origin" not in result.stdout:
        run_command(f"git remote add origin {remote_url}")
    else:
        print("Remote 'origin' already exists.")
    
    print(f"\nRepository set up successfully!")
    print(f"Remote URL: {remote_url}")
    print("\nTo push to GitHub, run:")
    print(f"git push -u origin master")
    print("\nMake sure you have created the repository on GitHub first at:")
    print(f"https://github.com/{username}/{repo_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up GitHub repository")
    parser.add_argument("--username", required=True, help="GitHub username")
    parser.add_argument("--repo_name", default="Hunyuan3D-Glasses", help="Repository name")
    parser.add_argument("--remote_url", help="Remote URL (if already created)")
    
    args = parser.parse_args()
    
    setup_github(args.username, args.repo_name, args.remote_url)
