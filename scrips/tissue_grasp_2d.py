# import sys
# from os.path import abspath, dirname, join
# sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'SOLAR_GP_master/scripts/test'))
# print(join(dirname(dirname(abspath(__file__))), 'SOLAR_GP_master/scripts/test'))
import SOLAR_API as SGP

from random import seed, random
import time

import math, taichi as ti, numpy as np
assert ti.__version__ > (0, 6, 7)

ti.init(ti.cuda)
ti.core.toggle_advanced_optimization(False)

def vec(*xs):
    return ti.Vector(xs)

dt = 0.01
Nx = 40
Ny = 4
Wx0 = 0.6
Wy0 = 0.06
L0 = Wy0 / Ny
K0 = 500
D0 = 0.05
G0 = 0.0005
A0 = 0.001
B0 = 0.2

pos = ti.Vector(2, ti.f32)
vel = ti.Vector(2, ti.f32)
attr_pos = ti.Vector(2, ti.f32)
attr_stren = ti.var(ti.f32)
ti.root.dense(ti.ij, (Nx, Ny)).place(pos, vel)
ti.root.place(attr_pos, attr_stren)

@ti.func
def reaction(I, J, k):
    ret = pos[I] * 0
    if J[0] < Nx and J[1] < Ny and all(J >= 0):
        dis = pos[I] - pos[J]
        ret = K0 * dis.normalized() * (k * L0 - dis.norm())
    return ret

@ti.kernel
def substep(is_grasped: ti.i32):
    # compute new accel and velocity
    for I in ti.grouped(pos):
        if I[0] < Nx - 1:
            if is_grasped and I[0] == 0 and I[1] == Ny - 1:
                continue
            acc = reaction(I, I + vec(0, 1), 1)
            acc += reaction(I, I - vec(0, 1), 1)
            acc += reaction(I, I + vec(1, 0), 1)
            acc += reaction(I, I - vec(1, 0), 1)
            acc += reaction(I, I + vec(1, 1), math.sqrt(2))
            acc += reaction(I, I - vec(1, 1), math.sqrt(2))
            acc += reaction(I, I + vec(1, -1), math.sqrt(2))
            acc += reaction(I, I - vec(1, -1), math.sqrt(2))
            acc[1] -= G0
            acc += attr_stren[None] * (attr_pos[None] - pos[I]).normalized()
            vel[I] *= ti.exp(-dt * D0)
            vel[I] += acc * dt

    # Collide with ground 
    for I in ti.grouped(pos):
        if pos[I][0] < 0:
            pos[I][0] = 0
            vel[I][0] = 0
        if pos[I][0] > 1:
            pos[I][0] = 1
            vel[I][0] = 0 
        if pos[I][1] < 0:
            pos[I][1] = 0
            vel[I][1] = 0 
        if pos[I][1] > 1:
            pos[I][1] = 1
            vel[I][1]  = 0

    # update position
    for I in ti.grouped(pos):
        pos[I] += vel[I] * dt


@ti.kernel
def init():
    for I in ti.grouped(pos):
        pos[I][0] = I[0] / Nx * Wx0 + (1-Wx0)/2
        pos[I][1] = I[1] / Ny * Wy0


def move2center():
    init()

    # # debug
    # print('pos shape: ', pos.to_numpy().shape)
    # print('top-left: ', pos.to_numpy()[0, Ny - 1])
    # print('top-left after reshape: ', pos.to_numpy().reshape(Nx*Ny, 2)[38, :])

    gui = ti.GUI('Move top-left to center')
    end = False
    is_pause = True
    ite_num = 0
    while True:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == gui.SPACE:
                is_pause = not is_pause
        if not end and not is_pause:
            for i in range(10):
                pos[0,Ny-1][0] += 3e-4
                pos[0,Ny-1][1] += 44e-5
                substep(1)

            ite_num += 1
            if ite_num == 100:
                end = True


        gui.circles(pos.to_numpy().reshape(Nx*Ny, 2), radius=1.8)
        gui.show()
        # if is_pause:
        #     time.sleep(10)
        #     is_pause = False

def get_exclude_points(Nx, Ny):
    exclude_points = [Ny - 1]
    for i in reversed(range(Ny)):
        exclude_points.append(Nx*Ny - i - 1)

    return exclude_points

def get_step_len(exclude_points):
    init()

    pos_np_init = pos.to_numpy().reshape(Nx*Ny, 2)
    pos_np_init = np.delete(pos_np_init, exclude_points, 0)

    for i in range(1000):
        pos[0,Ny-1][0] += 3e-4
        pos[0,Ny-1][1] += 44e-5
        substep(1)

    pos_np_final = pos.to_numpy().reshape(Nx*Ny, 2)
    pos_np_final = np.delete(pos_np_final, exclude_points, 0)

    step_len = (pos_np_final - pos_np_init)/1000
    return step_len.reshape(1, (Nx*Ny-1-Ny)*2)

def get_jitter(exclude_points):
    init()
    seed(1)

    cntls = np.zeros((15,2))
    fdbk_points = np.zeros((15, (Nx*Ny-1-Ny)*2))
    for i in range(15):
        cntls[i,:] = np.array([random(), random()])*1e-3
        pos[0, Ny - 1][0] += cntls[i,0]
        pos[0, Ny - 1][1] += cntls[i,1]
        substep(1) 

        pos_np = pos.to_numpy().reshape(Nx*Ny, 2)
        pos_np = np.delete(pos_np, exclude_points, 0)
        fdbk_points[i,:] = pos_np.reshape((Nx*Ny-1-Ny)*2,)
    
    return cntls, fdbk_points

if __name__ == '__main__':
    exclude_points = get_exclude_points(Nx, Ny)
    step_len = get_step_len(exclude_points)
    cntls, fdbk_points = get_jitter(exclude_points)

    t_start = SGP.tic()
    t_step = SGP.tic()

    # construct a SOLARGP to learn sin function
    inv_cntler = SGP.SOLARGP(num_idc=10, wgen=0.975,  max_num_models=3)

    # initialization
    inv_cntler.init(fdbk_points,cntls)

    init()
    gui = ti.GUI('Move top-left to center')
    end = False
    ite_num = 0
    while True:
        if not end:
            cntls_pred = np.zeros((20, 2))
            fdbk_train = np.zeros((20, (Nx*Ny-1-Ny)*2))
            for i in range(20):
                # predict
                cntls_pred[i,:] = inv_cntler.predict(step_len)

                pos[0, Ny - 1][0] += cntls_pred[i,0]
                pos[0, Ny - 1][1] += cntls_pred[i,1]
                substep(1)

                pos_np = pos.to_numpy().reshape(Nx*Ny,2)
                pos_np = np.delete(pos_np, exclude_points, 0)
                fdbk_train[i,:] = pos_np.reshape((Nx*Ny-1-Ny)*2,)

            ite_num += 1
            if ite_num == 50:
                end = True
                SGP.toc(t_start, 'inverse_controler_demo')

            # train global model
            inv_cntler.train_global(fdbk_train, cntls_pred, inv_cntler.mdrift, 20, use_old_Z=False)

            # patition
            inv_cntler.partition(fdbk_train, cntls_pred)

            # train local model
            inv_cntler.train_local()

            # timing 
            SGP.toc(t_step, '%d training'%(ite_num))
            t_step = SGP.tic()


        gui.circles(pos.to_numpy().reshape(Nx*Ny, 2), radius=1.8)
        gui.show()



# move2center()

# init()

# is_pause = False
# is_grasped = 0
# print('[Hint] LMB/RMB to attract/repel, MMB to set circle position')
# gui = ti.GUI('Mass-spring block')
# while True:
#     for e in gui.get_events(ti.GUI.PRESS):
#         if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
#             exit()
            
#         if e.key == ti.GUI.LMB:
#             if gui.is_pressed('Shift'):
#                 pos[0,Ny-1] = gui.get_cursor_pos()
#                 vel[0,Ny-1][0] = 0
#                 vel[0,Ny-1][1] = 0
#                 is_grasped = 1
#             else:
#                 is_grasped = 0
#                 attr_pos[None] = gui.get_cursor_pos()
#                 attr_stren[None] = A0
#         elif e.key == ti.GUI.RMB:
#             attr_pos[None] = gui.get_cursor_pos()
#             attr_stren[None] = -A0
#         elif e.key == gui.SPACE:
#             is_pause = not is_pause
#         else:
#             attr_stren[None] = 0
    
#     if not is_pause:
#         for i in range(200):
#             substep(is_grasped)
#     gui.circles(pos.to_numpy().reshape(Nx*Ny, 2), radius=1.8)
#     gui.show()