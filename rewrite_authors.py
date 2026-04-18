import subprocess
import sys

OLD_EMAIL = "121626257+neekunjchaturvedi@users.noreply.github.com"
NEW_NAME = "shafiankhan"
NEW_EMAIL = "mshafianak@gmail.com"

# Get all commits
result = subprocess.run(
    ["git", "log", "--format=%H", "HEAD"],
    capture_output=True, text=True
)
commits = result.stdout.strip().split("\n")
print(f"Found {len(commits)} commits to check")

# Check each commit's author
for sha in commits:
    r = subprocess.run(
        ["git", "log", "-1", "--format=%ae", sha],
        capture_output=True, text=True
    )
    email = r.stdout.strip()
    print(f"  {sha[:8]} -> {email}")
