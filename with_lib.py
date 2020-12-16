#!/bin/env python3

from pyeasyga import pyeasyga

# Некая точность (глубина) алгоритма, дальше будет очевидно, что она именно
# означает
# 
# ВАЖНО! Программа будет работать около минуты, это не зависание, все хорошо,
# нужно немного подождать. 
PRECISION = 3

# Мой вариант (по "новому" списку)
VARIANT = 18

# Данные из файла
data = []

# Здесь все очевидно. Единственный момент, что после первого readline
# внутренняя позиция потока чтения сдвинула вперед, поэтому можно продолжать
# читать циклически остальные данные. Таким образом мы читаем все сразу как
# надо, без дальнейшего удаления/изменения чего-то, что является весьма
# "дорогими" операциями (особенно удаление нулевого элемента из списка)
with open(f'data/{VARIANT}.txt') as f:
    arr = lambda string: [float(x) for x in string.split()]
    max_weight, max_size = arr(f.readline())
    for line in f:
        data.append(arr(line))

# Функция с официального сайта библиотеки, рекомендованная для задачи о
# многомерном рюкзаке
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

# Лучший получившийся вариант
best = {
    'cost': -1,
    'inclusive': None,
    'crossover': None,
    'mutation': None
}

# Создание самого алгоритма, все свойства по умолчанию из документации,
# только размер популяции 200 (вместно дефолтных 50)
ga = pyeasyga.GeneticAlgorithm(
    data,
    population_size=200,
    generations=100,
    crossover_probability=0.8,
    mutation_probability=0.2,
    elitism=True,
    maximise_fitness=True
)

ga.fitness_function = fitness 

# На качество результата наиболее сильное влияние оказывают вероятности
# скрещивания и мутации, именно их комбинации и перебираем в зависимости от
# заданной точности. Также каждая комбинация используется несколько раз, так
# как в алгоритме есть элемент случайности.
# 
# На самом деле, можно было бы при одних и тех же параметрах просто несколько
# раз выполнять алгоритм, и скорее всего был бы достигнут оптимум, так как
# элемент случайности оказывает на это достижение ключевую роль. Но все-таки
# я решил перебрать разные комбинации, хоть и по нескольку раз каждую.
for i in range(1, PRECISION + 1):
    ga.crossover_probability = i / PRECISION

    for j in range(1, PRECISION + 1):
        ga.mutation_probability = i / PRECISION
        
        for _ in range(1, PRECISION + 1):
            ga.run()
            cost, inclusive = ga.best_individual()

            if cost > best['cost']:
                best['cost'] = cost
                best['inclusive'] = inclusive
                best['crossover'] = i / PRECISION
                best['mutation'] = i / PRECISION

# Далее считываются нужные характеристики из полученных данных и формируется,
# выводится в консоль и записывается в файл результат

items = []
weight = 0
size = 0

for i in range(len(best['inclusive'])):
    if (best['inclusive'][i] != 1):
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
result += f'\nОбщая ценность: {int(best["cost"])}'
result += f'\nВероятностью мутаций: {best["crossover"]}'
result += f'\nВероятность скрещивания: {best["mutation"]}'

print(result)

with open('with_lib.txt', 'w') as f:
    f.write(result)