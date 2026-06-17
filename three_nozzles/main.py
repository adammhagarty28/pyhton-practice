nozzles= [
    {"name":"Nozzle A", "temperature":500, "status":"off"},
    {"name":"Nozzle B", "temperature":450, "status":"off"},
    {"name":"Nozzle C", "temperature":600, "status":"off"}
]

cooling_rate=20
room_temp=25

for step in range(30):
    for nozzle in nozzles:
        if nozzle["temperature"] > room_temp:
            nozzle["temperature"]=nozzle["temperature"]-cooling_rate
            nozzle["status"]="spraying"
        else:
            nozzle["status"]="off"

    print(f"---Step {step}---")
    for nozzle in nozzles:
        print(f"{nozzle['name']}: {nozzle['temperature']:.1f} C | {nozzle['status']}")                                  