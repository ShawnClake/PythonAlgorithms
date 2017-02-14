import random


def fitness(target, found, sym_count, length):
    if target == found:
        return 0
    modifier = (length - sym_count) / length
    if modifier <= 0 or modifier > 1:
        modifier = 1
    ratio = 1 / (target - found)
    #return ratio
    return ratio * modifier


def generate(length):
    return random.getrandbits(length * 4)


def symbol_lookup(bits):
    switcher = {
        0x0: 0,
        0x1: 1,
        0x2: 2,

        0x3: 3,
        0x4: 4,
        0x5: 5,
        0x6: 6,
        0x7: 7,
        0x8: 8,
        0x9: 9,
        0xa: '+',
        0xb: '-',
        0xc: '*',
        0xd: '/'
    }
    return switcher.get(bits, '')


def decode(bits, length):
    mask = 0xF
    sequence = []
    for x in range(0, length):
        sequence.append(symbol_lookup(bits & mask))
        bits = bits >> 4

    return sequence


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def calculate(sequence):
    # print(sequence)
    total = 0
    last_was_symbol = True
    symbol = ''
    symCount = 1
    for i in range(len(sequence)):
        # print("When symbol: ", sequence[i], " was encountered, the last sym was: ", last_was_symbol, ". The total is now: ", total)
        if sequence[i] == '':
            continue
        if last_was_symbol and not is_number(sequence[i]):
            continue
        if not last_was_symbol and is_number(sequence[i]):
            continue
        if symbol == '':
            total = sequence[i]
            symbol = 'zz'
            last_was_symbol = False
        elif not last_was_symbol:
            symbol = sequence[i]
            last_was_symbol = True
        else:
            if symbol == '+':
                total += sequence[i]
                symCount += 2
            elif symbol == '-':
                total -= sequence[i]
                symCount += 2
            elif symbol == '*':
                total *= sequence[i]
                symCount += 2
            elif symbol == '/':
                if sequence[i] == 0:
                    return False, 0
                total /= sequence[i]
                symCount += 2
            last_was_symbol = False
    return total, symCount


def mutate(bits, length):
    for x in range(0, length * 4):
        r = random.randint(1, 1000)
        if r == 576:
            bits ^= (1 << x)
            # print("A mutation has occurred.")
    return bits


def crossover(bitsa, bitsb, length):
    r = random.randint(1, length * 4)
    bitsa = (bitsa >> r) << r
    mask = 2**r - 1
    bitsb &= mask
    return bitsa | bitsb


def roulette(total_fitness, population):
    fitness_level = 0
    for key, value in population.items():
        fitness_level += value
        if fitness_level >= total_fitness:
            return key


def best_genome(population):
    highest_fitness = 0
    genome = {}
    for key, value in population.items():
        if float(value) > float(highest_fitness):
            genome.clear()
            highest_fitness = value
            genome[hex(key)] = value
        elif float(value) == float(highest_fitness):
            genome[hex(key)] = value
    # print("This generations best genomes were: ", genome)

# The starting point of the program
def main():
    length = 64
    target = 511
    pop_size = 50
    population = {}
    final_bits = ''
    running = True
    generation = 0
    timeout = False
    final_sym_count = 0
    sum = 0
    avg_count = 0

    nuke = length * pop_size
    ping_avg = (nuke / 5) - 1

    while running:
        total_fitness = 0

        generation += 1
        if generation % 1000 == 0:
            print("Generation: ", generation)

        if generation % nuke == 0:
            population.clear()
            print("Population was nuked and respawned")
            avg_count = 0
            sum = 0

        if generation % ping_avg == 0 and avg_count != 0:
            print("Current fitness avg: ", (sum / avg_count))

        # print(len(population))

        while len(population) < pop_size:
            population[generate(length)] = 0

        for key, value in population.items():
            # print("Genome is: ", hex(key))
            decoded = decode(key, length)
            calculated, sym_count = calculate(decoded)
            # print("Genome value is: ", calculated)
            if calculated == target:
                # print(calculated)
                running = False
                final_bits = key
                final_sym_count = sym_count
                break
            fitnessed = fitness(target, calculated, sym_count, length)
            population[key] = fitnessed
            total_fitness += fitnessed
            sum += fitnessed
            avg_count += 1

        if generation > 100000:
            running = False
            timeout = True
            break

        if running:
            best_genome(population)

        adders = {}

        for i in range(pop_size):
            if not running:
                break
            bitsa = ''
            bitsb = ''
            trials = 0
            while bitsa == bitsb and trials < 10:
                r1 = random.uniform(0, total_fitness)
                r2 = random.uniform(0, total_fitness)
                bitsa = roulette(r1, population)
                bitsb = roulette(r2, population)
                trials += 1

            bitsc = crossover(bitsa, bitsb, length)
            bitsc = mutate(bitsc, length)
            adders[bitsc] = 0

        population.clear()

        population = adders.copy()

        adders.clear()

    if timeout:
        print("Genetic algorithm ended without a solution as we passed 100,000 generations")
    else:
        print("Solution found with genome: ", hex(final_bits))
        print("Solution found symbol count: ", final_sym_count)
        print("The solution was found in generation: ", generation)

main()
