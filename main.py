from hparam_optimizer import compare

if __name__ == '__main__':
    k = 1
    for _ in range(10):
        print(f"{k} Iteration: " "\n")
        compare()
        k += 1
