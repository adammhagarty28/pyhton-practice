height = 10.0
velocity = 0.0
gravity = -1.0

for step in range(20):
    velocity = velocity + gravity
    height = height + velocity

    if height < 0:
        height = 0.0
        velocity = -velocity

    print(f"Step {step}: height = {height:.2f}")