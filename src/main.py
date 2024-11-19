import time

def main():
    print("RAG Router system starting...")
    try:
        while True:
            print("System running and waiting for queries...")
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()