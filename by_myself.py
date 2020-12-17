#!/bin/env python3

'''
Вариант 18
 - Начальная популяция: случайная генерация
 - Отбор: выбор каждой особи пропорционально приспособленности (рулетка)
 - Скрещивание: многоточечный с 3мя точками
 - Мутация: добавление 1 случайной вещи 5% особей
 - Новая популяция: "штраф" за "старость" 10% функции приспособленности,
   выбор лучших
'''

from random import randint, random

class GeneticAlgorithm:
    '''Класс алгоритма. Реализован по аналогии с библиотеками.'''

    def __init__(
        self,
        data,
        *,
        fitness,
        population_size=200,
        generations=500,
        mutation_proportion=0.05,
        age_penalty=0.1
    ):
        '''Конструктор сохраняет все внешние данные, которые могут понадобится
        на том или ином этапе алгоритма, в экземпляре класса. Принимаемые
        параметры:

        data
            Исодные данные, должны быть списком или кортежем, при этом неважно,
            что именно это за список, алгоритм работает полностью
            абстрагировавшись от данных, единственное, с чем данные должны
            согласовываться, так это с фитнес-функцией.
        fitness
            Фитнес-функция, вычисляющая для каждого решения (особи генетического
            алгоритма) некое значение, по которому можно судить о его
            оптимальности и сравнивать с другими.
            Эта функция обратного вызова должна первым параметром принимать
            особь (массив генов, то есть нулей и единиц), а вторым данные, на
            которых работает алгоритм, и возвращать неотрицательное число,
            причем 0 означает, что данная особь некорректна для исходных данных.
        population_size
            Количество особей в популяции, по умолчанию 200.
        generations
            Количество поколений, с которыми алгоритм будет работать прежде, чем
            вернуть результат, по умолчанию 500.
        mutation_proportion
            Доля особей в каждой популяции (кроме начальной), которые будут
            подвержены мутации, по умолчанию 0.05 (5%).
        age_penalty
            "Штраф" за "старость" которые получают особи из старой популяции при
            формировании новой, по умолчанию 0.2 (20%).
        '''

        self.__data = data
        self.__population_size = population_size
        self.__generations = generations
        self.__mutation_proportion = mutation_proportion
        self.__age_penalty = age_penalty
        self.__fitness = fitness
        self.__gene_num = len(data) # Количество генов
    
    def __random_individual(self):
        '''Создание случайной особи. Особь представляет из себя массив из нулей
        и единиц длинный равной количеству генов, то есть количеству элементов в
        исходном массиве данных.
        
        Единица значит, что соответствующий (по индексу) элемент из исходного
        массива данных входит в данный набор, а ноль, что не входит.'''

        individual = [randint(0, 1) for _ in range(self.__gene_num)]

        while (self.__fitness(individual, self.__data) == 0):
            individual = [randint(0, 1) for _ in range(self.__gene_num)]
        
        return individual

    def __random_population(self):
        '''Создание случайной популяции. Случайная популяция, это просто
        популяция из случайных особей.'''

        return [
            self.__random_individual() for _ in range(self.__population_size)
        ]

    def __select_individuals_indices(self, population, N=None):
        '''Отбор из данной популяции особей пропорционально их
        приспособленности. Параметры:
        
        population
            Популяция, в которой нужно произвести отбор.
        N
            Количество особей, которое нужно отобрать. Если этот параметр не
            задан, то будет отобрано особей количеством равным размеру
            популиции, при этом некоторые особи в результирующем наборе могут
            втретиться несколько раз, это особенность пропорционального
            отбора.
        
        Возвращается массив индексов отобранных особей из исходной популяции.'''
        
        if (N == None):
            N = self.__population_size
        
        selected = []
        probabilities = []
        values_sum = 0

        for i in range(len(population)):
            value = self.__fitness(population[i], self.__data)
            probabilities.append([i, value])
            values_sum += value

        probabilities.sort(key=lambda x: x[1])

        probabilities[0][1] /= values_sum

        for i in range(1, len(probabilities)):
            probabilities[i][1] /= values_sum
            probabilities[i][1] += probabilities[i - 1][1]

        probabilities[-1][1] = 1.0

        for _ in range(N):
            r = random()

            for i, p in probabilities:
                if r >= p:
                    continue
                selected.append(i)
                break
        
        return selected
    
    # Алгоритм можно реализовать для любого произвольного количества точек с
    # помощью цикло, но сейчас не не суть как важно.
    def __cross_individuals(self, first, second):
        '''Скрещевание двух особей. Скрещивание происходит многоточечным методом
        с тремя точками. На вход принимаются две особи родителя, и возвращаются
        две особи потомка.'''
        points = []

        while len(points) != 3:
            point = randint(0, self.__gene_num - 1)
            if point not in points:
                points.append(point)
        
        points.sort()

        first_child = (first[0:points[0]] +
            second[points[0]:points[1]] +
            first[points[1]:points[2]] +
            second[points[2]:self.__gene_num])
        
        second_child = (second[0:points[0]] +
            first[points[0]:points[1]] +
            second[points[1]:points[2]] +
            first[points[2]:self.__gene_num])
        
        return first_child, second_child
    
    # Предполагается, что все данные хорошие, при правильном использовании
    # так и будет, но если количество особей в популяции будет нечетным,
    # то здесь возникнет бесконечный цикл. Это далеко не единственный момент,
    # где стоило бы сделать дополнительные проверки, но я от этого воздержался,
    # так как задание все же не про это. 
    def __crossingover(self, population):
        '''Кроссинговер по какой-то популяции (набора особей). Каждая особь
        из набора будет скрещена ровно один раз. Выбор пар производится
        случайным образом. На вход принимается массив особей, а возращается
        массив их потомков (тоже особей).'''

        crossed = [False for _ in population]
        children = []

        # Получение еще не скрещенной особи (но она сразу становится таковой)
        def get_not_crossed():
            i = randint(0, len(population) - 1)
            while crossed[i] == True:
                i = randint(0, len(population) - 1)
            crossed[i] = True
            return i

        while False in crossed:
            i = get_not_crossed()
            j = get_not_crossed()
            
            children.extend(
                self.__cross_individuals(population[i], population[j])
            )
        
        return children
    
    def __mutate_individual(self, individual):
        '''Мутация одной особи. Мутация заключется в дбавлении одной случайной
        вещи, значит нужно просто выбрать слуйную ячейку в масиве особи, где
        стоит записан ноль и записать туда единицу.'''

        i = randint(0, self.__gene_num - 1)
        while individual[i] == 1:
            i = randint(0, self.__gene_num - 1)
        individual[i] = 1
    
    def __mutate_population(self, population, *, proportion=None):
        '''Мутация популяции особей. А если быть точнее, то ее части, которая
        задается в долях параметром proportion.'''

        if proportion == None:
            proportion = self.__mutation_proportion

        n = round(len(population) * proportion)
        mutated = []

        for _ in range(n):
            i = randint(0, len(population) - 1)
            while i in mutated:
                i = randint(0, len(population) - 1)
            mutated.append(i)
            self.__mutate_individual(population[i])
        
    def best_individual(self):
        '''Непосредственное применение алгоритма. Возвращается кортеж, в котором
        на первой позиции найденное максимальное значение, а на второй особь, у
        которой оно достигается.'''

        # Выбор случайной начальной популяции
        population = self.__random_population()

        for _ in range(self.__population_size):
            # Отбор особей для скрещевания
            indecies_for_crossingover = self.__select_individuals_indices(
                population
            )

            population_for_crossingover = [
                population[i] for i in indecies_for_crossingover
            ]

            # Кроссинговер
            new_generation = self.__crossingover(population_for_crossingover)

            # Мутации
            self.__mutate_population(population)
            self.__mutate_population(new_generation)

            # "Штраф" за "старость" при формировании новой популяции
            new_population = [
                (
                    self.__fitness(individual, self.__data) *
                        (1 - self.__age_penalty),
                    individual
                )
                for individual in population
            ]

            new_population += [
                (self.__fitness(individual, self.__data), individual)
                for individual in new_generation
            ]
            
            new_population.sort(reverse=True)
            population = new_population[0:self.__population_size]

            # На самом деле, нигде не делалась проверка на то, что после
            # скрещевания или мутации получилась "правбильная" особь, то есть
            # не вызывалась фитнес функция, чтобы отбросить лишнее. Но на самом
            # деле в этом необходимости нет, так как если какая-то особь не
            # удовлетворяет исходным данным, для нее фитнес-функция вернет ноль.
            # А зате в процессе сортировки она попадет в конец списка и в любом
            # случае будет обрезана.
            
            population = [individual for value, individual in population]
        
        return (self.__fitness(population[0], self.__data), population[0])

# Далее код почти один в один сопрадает с тем, что был в предудыщем пункте
# задания, раз что немного он сптал попроще.

# Мой вариант (по "новому" списку)
VARIANT = 18

# Данные из файла
data = []

with open(f'data/{VARIANT}.txt') as f:
    arr = lambda string: [float(x) for x in string.split()]
    max_weight, max_size = arr(f.readline())
    for line in f:
        data.append(arr(line))


# Фитнес функция
def fitness(individual, data):
    weight, volume, price = 0, 0, 0
    for (selected, item) in zip(individual, data):
        if selected:
            weight += item[0]
            volume += item[1]
            price += item[2]
    if weight > max_weight or volume > max_size:
        price = 0
    return price

ga = GeneticAlgorithm(data, fitness=fitness)
cost, individual = ga.best_individual()

items = []
weight = 0
size = 0

for i in range(len(individual)):
    if (individual[i] != 1):
        continue

    weight += data[i][0]
    size += data[i][1]

    items.append('Позиция #{:>2}: {:>4} {:>3} {:>3}'.format(
        i + 1, int(data[i][0]), data[i][1], int(data[i][2])
    ))

result = f'ВАРИАНТ {VARIANT}\n\n'
result += '\n'.join(items)
result += f'\n\nОбщий вес: {int(weight)} / {int(max_weight)}'
result += f'\nОбщий размер: {size} / {int(max_size)}'
result += f'\nОбщая ценность: {int(cost)}'

print(result)

with open('by_myself.txt', 'w') as f:
    f.write(result)