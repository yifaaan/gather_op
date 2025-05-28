import random

def three_d():
    with open('../128_128_128.txt', 'w') as f:
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    # Calculate the number (can be customized based on what numbers you want)
                    num = random.uniform(-1, 1)
                    f.write(f"{num}\n")

    with open('../indices_128_3d.txt', 'w') as f:
        for i in range(128):
                    # Calculate the number (can be customized based on what numbers you want)
                    num = random.randint(0, 127)
                    f.write(f"{num}\n")
                    
    with open('../indices_72_3d.txt', 'w') as f:
        for i in range(72):
                    # Calculate the number (can be customized based on what numbers you want)
                    num = random.randint(0, 127)
                    f.write(f"{num}\n")

def four_d():
    with open("../128_128_128_5.txt", "w") as f:
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    for l in range(5):
                        num = random.uniform(-1, 1)
                        f.write(f"{num}\n")

    with open("../indices_128_4d.txt", "w") as f:
        for i in range(128):
                    # Calculate the number (can be customized based on what numbers you want)
                    num = random.randint(0, 127)
                    f.write(f"{num}\n")

def five_d():
    with open("../5_128_3_128_128.txt", "w") as f:
        for i in range(5):
            for j in range(128):
                for k in range(3):
                    for l in range(128):
                        for m in range(128):
                            num = random.uniform(-1, 1)
                            f.write(f"{num}\n")

    with open("../indices_128_5d.txt", "w") as f:
        for i in range(128):
            # Calculate the number (can be customized based on what numbers you want)
            num = random.randint(0, 127)
            f.write(f"{num}\n")

if __name__ == "__main__":
    three_d()
    four_d()
    five_d()