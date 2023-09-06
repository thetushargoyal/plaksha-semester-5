# Define the state representation as a tuple (B, N, R)
# B: Number of bacteria
# N: Number of neutrophils
# R: Level of bacterial resistance (an integer from 0 to 5)

# Define the move generation (neighborhood) function
def movegen(state):
    neighbors = []

    # Rule 1: Bacteria Reproduction
    if state[0] > 0:
        new_state = (state[0] + 1, state[1], state[2])
        neighbors.append(new_state)

    # Rule 2: Neutrophil Attack
    if state[0] > 0 and state[1] > 0 and state[2] < 5:
        new_state = (state[0] - 1, state[1] - 1, state[2])
        neighbors.append(new_state)

    # Rule 3: Neutrophil Migration
    if state[1] > 0:
        new_state = (state[0], state[1] - 1, state[2])
        neighbors.append(new_state)

    # Rule 4: Bacteria Die
    if state[0] > 0:
        new_state = (state[0] - 1, state[1], state[2])
        neighbors.append(new_state)

    # Rule 5: Bacterial Resistance Development
    if state[2] < 5 and state[0] > 0:
        new_state = (state[0], state[1], state[2] + 1)
        neighbors.append(new_state)

    return neighbors

# Define the goal test function
def goaltest(state):
    # Check if all bacteria are eliminated (B == 0)
    return state[0] == 0

# Example usage:
initial_state = (1, 0, 5)
print("Initial State:", initial_state)
print("Goal Test Result:", goaltest(initial_state))
print("Neighbors:", movegen(initial_state))
