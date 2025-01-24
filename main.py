from drone_agent import DroneAgent

def main():
    try:
        drone = DroneAgent()
        drone.run()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()