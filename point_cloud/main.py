points=[
    {"x":0, "y":0, "temperature":500},
    {"x":1, "y":0, "temperature":480},
    {"x":2, "y":0, "temperature":460},
    {"x":0, "y":1, "temperature":470},
    {"x":1, "y":1, "temperature":450}
]

cooling_rate=10
room_temp=25

for step in range(10):
    for point in points:
        if point["temperature"] > room_temp:
            point["temperature"]= point["temperature"]-cooling_rate

    print(f"---Step {step}---")
    for point in points:
        print(f"({point['x']}, {point['y']}): {point['temperature']:.1f} C")