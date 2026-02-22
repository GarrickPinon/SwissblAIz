
import requests
import time
import sys

SERVER_URL = "https://cactusevals.ngrok.app"
HEADERS = {"ngrok-skip-browser-warning": "true"}
SUBMISSION_ID = "bb9c3056c15e417390a04ed3c3c5826a"

def monitor():
    print(f"Monitoring submission: {SUBMISSION_ID}")
    last_progress = ""
    while True:
        try:
            resp = requests.get(
                f"{SERVER_URL}/eval/status",
                params={"id": SUBMISSION_ID},
                headers=HEADERS,
                timeout=10
            )
            if resp.status_code != 200:
                print(f"Error {resp.status_code}. Retrying...")
                time.sleep(5)
                continue
            
            status = resp.json()
            if status["progress"] and status["progress"] != last_progress:
                last_progress = status["progress"]
                print(f"  [{status['progress']}]")

            if status["status"] == "complete":
                result = status["result"]
                print("\n" + "=" * 50)
                print(f"  RESULTS for team '{result['team']}'")
                print("=" * 50)
                print(f"  Total Score : {result['score']:.1f}%")
                print(f"  Avg F1      : {result['f1']:.4f}")
                print(f"  Avg Time    : {result['avg_time_ms']:.0f}ms")
                print(f"  On-Device   : {result['on_device_pct']:.0f}%")
                print("=" * 50)
                return

            if status["status"] == "error":
                print(f"\nError: {status.get('error', 'Unknown error')}")
                return

            if status['status'] == 'queued':
                print(f"  Queued (position: {status.get('queue_size', '?')})...", end="\r", flush=True)

        except Exception as e:
            print(f"\nConnection error: {e}. Retrying...")
        
        time.sleep(10)

if __name__ == "__main__":
    monitor()
