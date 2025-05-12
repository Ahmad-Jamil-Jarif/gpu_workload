import numpy as np

class GPU:
    def __init__(self, num_sms, threads_per_block, global_mem_latency, shared_mem_latency):
        self.num_sms = num_sms
        self.threads_per_block = threads_per_block
        self.global_mem_latency = global_mem_latency
        self.shared_mem_latency = shared_mem_latency
        self.cycles = 0
        self.global_memory_grid = None
        self.shared_memory_grid = None

    def simulate_workload(self, A, B, C):

        rows, cols = A.shape[0], B.shape[1]
        latency = 0

        self.global_memory_grid = np.zeros_like(A)
        self.shared_memory_grid = np.zeros_like(A)

        for row in range(rows):
            for col in range(cols):
                C[row, col] = 0
                for k in range(A.shape[1]):

                    self.global_memory_grid[row, k] += 1
                    self.global_memory_grid[k, col] += 1
                    self.shared_memory_grid[row, k] += 1

                    latency += self.global_mem_latency + self.shared_mem_latency
                    C[row, col] += A[row, k] * B[k, col]
                    latency += 1

        self.cycles = latency
        return C, latency

    def display_memory_access(self):
        """
        Prints memory access patterns for global and shared memory.
        """
        print("\nGlobal Memory Access Pattern (Access Count):")
        for row in self.global_memory_grid:
            print(" ".join(f"{int(val):3}" for val in row))

        print("\nShared Memory Access Pattern (Access Count):")
        for row in self.shared_memory_grid:
            print(" ".join(f"{int(val):3}" for val in row))


def input_matrix(name):
    rows = int(input(f"Enter the number of rows for matrix {name}: "))
    cols = int(input(f"Enter the number of columns for matrix {name}: "))
    print(f"Enter the elements for matrix {name} (row by row):")
    elements = []
    for _ in range(rows):
        elements.append(list(map(int, input().split())))
    return np.array(elements)

if __name__ == "__main__":
    print("Welcome to the GPU Simulation Program!")

    A = input_matrix("A")
    B = input_matrix("B")

    if A.shape[1] != B.shape[0]:
        print("Error: The number of columns in A must match the number of rows in B.")
    else:

        num_sms = int(input("Enter the number of Streaming Multiprocessors (SMs): "))
        threads_per_block = int(input("Enter the number of threads per block: "))
        global_mem_latency = int(input("Enter the global memory latency (in cycles): "))
        shared_mem_latency = int(input("Enter the shared memory latency (in cycles): "))

        gpu = GPU(num_sms, threads_per_block, global_mem_latency, shared_mem_latency)

        C = np.zeros((A.shape[0], B.shape[1]))

        result, total_cycles = gpu.simulate_workload(A, B, C)

        print("\nMatrix A:")
        print(A)
        print("\nMatrix B:")
        print(B)
        print("\nResultant Matrix C:")
        print(result)
        print(f"\nTotal Execution Cycles: {total_cycles}")

        gpu.display_memory_access()
