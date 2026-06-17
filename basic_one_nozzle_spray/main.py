import nozzle

temperature=500
cooling_rate=20
room_temp=25

for step in range (20):
    if temperature > room_temp:
        temperature=temperature - cooling_rate
        status="spraying"
    else:
        status="off"
    message=nozzle.describe_nozzle(temperature,status)
    print(f"Step {step}:{message}")
    