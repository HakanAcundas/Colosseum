#AUTHORS : HAKAN ACUNDAŞ , BATIKAN ÖZKUL , YUNUS EMRE SANCAK


import pygame, math
import random
import numpy as np
import torch
import torch.nn as nn
from random import randrange

pygame.init()

display_width = 1000
display_height = 600

gameExit = False
gameTimer = 0
clock = pygame.time.Clock()
FPS = 30

dash_cooldown = FPS * 2
match_cooldown = FPS * 20

# Neccesssary variables for gladiators

GLAD_SPRITE = pygame.image.load("Glad.PNG")


gameTimer = 0
move_power = 4
dash_power = 8
border_power = 4
bounce_power = 5
angle_rate = 3
width = 72
heigth = 72


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(6, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 8)
        self.fc4 = nn.Linear(8, 6)
        self.ol1 = nn.Linear(6, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        out = self.ol1(out)
        out = self.sigmoid(out)
        return out

    def get_biases(self):
        fc1b = self.fc1.bias.data.numpy()
        fc2b = self.fc2.bias.data.numpy()
        fc3b = self.fc3.bias.data.numpy()
        fc4b = self.fc4.bias.data.numpy()
        fcolb = self.ol1.bias.data.numpy()
        return fc1b, fc2b, fc3b, fc4b, fcolb

    def get_weigths(self):
        fc1w = self.fc1.weight.data.numpy()
        fc2w = self.fc2.weight.data.numpy()
        fc3w = self.fc3.weight.data.numpy()
        fc4w = self.fc4.weight.data.numpy()
        fcolw = self.ol1.weight.data.numpy()
        return fc1w, fc2w, fc3w, fc4w, fcolw

    def set_biases(self, mre):
        with torch.no_grad():
            self.fc1.bias = nn.Parameter(torch.Tensor(mre[0]))
            self.fc2.bias = nn.Parameter(torch.Tensor(mre[1]))
            self.fc3.bias = nn.Parameter(torch.Tensor(mre[2]))
            self.fc4.bias = nn.Parameter(torch.Tensor(mre[3]))
            self.ol1.bias = nn.Parameter(torch.Tensor(mre[4]))

    def set_weights(self, mra):
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.Tensor(mra[0]))
            self.fc2.weight = nn.Parameter(torch.Tensor(mra[1]))
            self.fc3.weight = nn.Parameter(torch.Tensor(mra[2]))
            self.fc4.weight = nn.Parameter(torch.Tensor(mra[3]))
            self.ol1.weight = nn.Parameter(torch.Tensor(mra[4]))


class Gladiator(pygame.sprite.Sprite):
    def __init__(self, x, y, NN=None):
        if (NN == None):
            self.NN = Model()

        else:
            self.NN = NN

        pygame.sprite.Sprite.__init__(self)

        self.x = x
        self.y = y
        self.a = 0
        self.v = 0
        self.f = 1
        self.cooldown_tracker = 0
        self.is_Dash = False
        self.is_bounce = False
        self.angle = 0
        self.fitness = 0
        self.genes = []

    def decide(self, input):
        # ---Cooldown---#
        if self.cooldown_tracker != 0:
            self.cooldown_tracker += 1

            if self.cooldown_tracker > dash_cooldown:
                self.cooldown_tracker = 0
        # -----Input-----#
        kararlar = self.NN(input)
        # -----Decisions----#
        if kararlar[0][0] > 0.5 and self.cooldown_tracker == 0:
            self.cooldown_tracker += 1
            self.a = dash_power
            self.is_Dash = True

        desiciontorot = kararlar[0][1] - kararlar[0][2]
        self.rotation(desiciontorot.item())

        if kararlar[0][3] > 0.5:
            self.move()

    def rotation(self, rotateval):
        # saat yönü
        if rotateval > 0.05:
            self.angle -= angle_rate
        # saat yönü tersi
        if rotateval < -0.05:
            self.angle += angle_rate
        self.angle = self.angle % 360

    def dash(self):
        if self.is_Dash:
            rate_x = math.cos(math.radians(self.angle))
            rate_y = -math.sin(math.radians(self.angle))
            self.x += rate_x * self.v
            self.y += rate_y * self.v
            self.a -= self.f
            self.v += self.a
            if self.a == -dash_power + 1:
                self.is_Dash = False
                self.v = 0
                self.a = 0

    def move(self):
        if not self.is_Dash:
            if self.v < move_power:
                self.v = move_power
            rate_x = math.cos(math.radians(self.angle))
            rate_y = -math.sin(math.radians(self.angle))
            self.x += rate_x * self.v
            self.y += rate_y * self.v
            self.v -= self.f
            self.v += self.a

    def border_col(self):
        if self.x < 32 and 90 < self.angle < 270:
            self.x += self.v 
            self.angle += 180 - self.angle * 2
        if self.x > display_width - 36 and (270 < self.angle or self.angle < 90):
            self.x -= self.v 
            self.angle += 180 - self.angle * 2

        if self.y < 32 and 0 < self.angle < 180:
            self.y += self.v
            self.angle += 360 - self.angle * 2
        if self.y > display_height - 36 and 180 < self.angle < 360:
            self.y -= self.v
            self.angle += 360 - self.angle * 2


    def glad_col(self):
        if self.is_bounce:
            rate_x = math.cos(math.radians(self.angle))
            rate_y = -math.sin(math.radians(self.angle))
            if self.a <= -bounce_power:
                self.is_bounce = False
                self.v = 0
                self.a = 0
            self.a -= self.f
            back_v = -self.v * 2
            self.x += rate_x * back_v
            self.y += rate_y * back_v


def inputmaker(Glad1, Glad2):
    global display_width
    global display_height
    g1x = Glad1.x / display_width
    g2x = Glad2.x / display_width
    g1y = Glad1.y / display_height
    g2y = Glad2.y / display_height
    g1_ang = Glad1.angle / 360
    g2_ang = Glad2.angle / 360

    input = torch.FloatTensor([[g1x, g2x, g1y, g2y, g1_ang, g2_ang]])
    return input


def distancecalculator(Glad1, Glad2):
    distance = math.sqrt(((Glad1.x - Glad2.x) ** 2) + ((Glad1.y - Glad2.y) ** 2))
    return distance
    


def collision(g1, g2):
    global gameExit
    rate_x = math.cos(math.radians(g1.angle))
    rate_y = -math.sin(math.radians(g1.angle))
    rate_xx = math.cos(math.radians(g2.angle))
    rate_yy = -math.sin(math.radians(g2.angle))
    alphaDiff = ((abs(g1.angle - g2.angle) % (2 * 180)) / 180) * 180
    alphaDiff = alphaDiff % 360
    d = math.sqrt(pow(g1.x - g2.x, 2) + pow(g1.y - g2.y, 2))
    x_diff = g1.x - g2.x
    y_diff = g1.y - g2.y
    if d < 57:
        if alphaDiff < 90 or alphaDiff > 270:
            if x_diff < 15 and y_diff < 15 and not (270 < g1.angle < 360):  # 2.Bölge
                g2.fitness += 10
                print("2nd win")
            elif x_diff >= 15 and y_diff <= 15 and not (180 < g1.angle < 300):  # 1.Bölge
                print("2nd win")
                g2.fitness += 10
            elif x_diff < 15 and y_diff > 15 and not (0 < g1.angle < 110):  # 3.Bölge
                print("2nd win")
                g2.fitness += 10
            elif x_diff > 15 and y_diff > 15 and not (75 < g1.angle < 200):  # 4.Bölge
                print("2nd win")
                g2.fitness += 10
            else:
                print("1st win")
                g1.fitness += 10
               
            isHitting = True
        else:
            isHitting = False
            g1.fitness += 1
            g2.fitness += 1

        if isHitting:
            gameExit = True
            
        else:
            g1.x -= rate_x * 10
            g1.y -= rate_y * 10
            g2.x -= rate_xx * 10
            g2.y -= rate_yy * 10
            g1.angle += 180
            g2.angle += 180


class evolution:
    def __init__(self, max_popu=200):
        self.max_popu = max_popu
        self.population = []
        for i in range(max_popu):
            if i == 0:
                g1 = Gladiator(display_width / 2, display_height / 3)
                self.population.append(g1)
            elif i == 1:
                g1 = Gladiator(display_width / 2, 2 * (display_height / 3))
                self.population.append(g1)
            elif i % 2 == 0:
                g1 = Gladiator(display_width / 2, display_height / 3)
                self.population.append(g1)
            else:
                g1 = Gladiator(display_width / 2, 2 * (display_height / 3))
                self.population.append(g1)

    def genetic(self):
        new_population = []
        fitness_dict = {g: g.fitness for g in self.population}
        champ_list = self.population
        champ_list.sort(key=lambda x: x.fitness, reverse=True)
        
        print("Average Fitness:", sum(fitness_dict.values())/len(fitness_dict))
        print("Max Value: " , max(fitness_dict.values()))
        print("Min Value: " , min(fitness_dict.values()))


        # storing champions
        champs = champ_list[:10]
        for g in champs:
            g.fitness = 0

        for i in range(self.max_popu - len(champs)):
            parent = random.choices(self.population, fitness_dict.values(), k=2)

            # feeding the crossover
            p1 = parent[0]
            p2 = parent[1]

            # crossover
            child_biases = np.concatenate((p1.NN.get_biases()[:3], p2.NN.get_biases()[3:]))
            child_weights = np.concatenate((p1.NN.get_weigths()[:3], p2.NN.get_weigths()[3:]))
            if np.random.rand() < 0.1:
                child_weights = self.mutation(child_weights)
            child_NN = Model()
            child_NN.set_weights(child_weights)
            child_NN.set_biases(child_biases)
            if i % 2 == 0:
                child_traveler = Gladiator(display_width / 2, display_height / 3, child_NN)
            else:
                child_traveler = Gladiator(display_width / 2, 2 * (display_height / 3), child_NN)
            new_population.append(child_traveler)
        new_population.extend(champs)
        self.population = new_population
        

    # bit flip mutation
    def mutation(self, gene):
        col = randrange(len(gene))
        neuron = randrange(len(gene[col]))
        gene[col][neuron] = torch.rand(len(gene[col][neuron]))
        return gene


# ------------------------------------------------------------------------------------
def evolve():
    ev1 = evolution()
    for i in range(200):
        print(i)
        if i >= 1:
            ev1.genetic()

        for j in range(ev1.max_popu):
            if j % 2 == 0:
                g1 = ev1.population.pop(0)
                g2 = ev1.population.pop(0)
                gameLoop(g1, g2)
                ev1.population.append(g1)
                ev1.population.append(g2)
            else:
                continue


def gameLoop(g1, g2, fps=10000000000000000):
    global gameExit
    gameExit = False
    gameTimer = 0
    # global self.cooldown_tracker_nn

    while not gameExit:
        gameTimer += 1

        if gameTimer > match_cooldown:
            distance = distancecalculator(g1, g2)
            g1.fitness += 1 / int(distance) * 3
            g2.fitness += 1 / int(distance) * 3
            gameTimer = 0
            gameExit = True

        
        inputgec = inputmaker(g1, g2).float()
        inputgec2 = inputmaker(g2, g1).float()
        
        collision(g1, g2)
        g1.decide(inputgec)
        g2.decide(inputgec2)
        g1.dash()
        g2.dash()
        g1.border_col()
        g2.border_col()
        clock.tick(fps)
    gameExit = False


evolve()

