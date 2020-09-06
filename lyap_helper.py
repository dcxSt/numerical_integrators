from n_body import *
import copy
import matplotlib.pyplot as plt
import random
import math
import numpy as np

# helper functions for finding canonical p,q coords of planets
def rad(theta):
    # converts from degrees to radians
    return theta * math.pi / 180

def polar_to_cartesian(r,se_lat,se_lon):
    x = r*math.cos(rad(se_lat)) * math.cos(rad(se_lon))
    y = r*math.cos(rad(se_lat)) * math.sin(rad(se_lon))
    z = r*math.sin(rad(se_lat))
    return np.array([x,y,z])

def kg_to_100earth_masses(m):
    return m / (5.972 * 10**26)

def get_p(mass,xyz1,xyz2,days=1):
    return 100 * mass * (xyz2-xyz1)/days # units are earthmasses AU / day

# information from wikipedia and https://omniweb.gsfc.nasa.gov/coho/helios/heli.html 
# The (rough) positions and momenta on 1st of jan 2000
sun = {"name":"sun","mass":3330.54,"coords":np.array([0.0,0.0,0.0,0.0,0.0,0.0])}
mercury = {"name":"mercury","mass":0.0005527,"coords":np.array([-0.130,-0.450,-0.0250,1.219e-03,-3.47e-04,-1.36e-04])}
venus = {"name":"venus","mass":0.008150,"coords":np.array([-0.7133,-0.073093,0.04019,1.664e-03,-0.016469,-4.09e-04])}
earth = {"name":"earth","mass":0.01,"coords":np.array([-0.1936,0.9603,0.0,-0.01676,-3.38e-03,0.0])}
mars = {"name":"mars","mass":0.0010745,"coords":np.array([1.388,0.04726,-0.03275,-5.37e-05,1.6147e-03,5.21e-05])}
jupiter = {"name":"jupiter","mass":3.1781,"coords":np.array([3.98907,2.96254,-0.10408,-1.6433,2.212,0.0])}
saturn = {"name":"saturn","mass":0.9516,"coords":np.array([6.3717,6.5981,-0.36841,-0.41095,0.39685,0.0])}
uranus = {"name":"uranus","mass":0.145361,"coords":np.array([14.4841,-13.6730,-0.2434,0.04336,0.04593,0.0])}
neptune = {"name":"neptune","mass":0.1715,"coords":np.array([16.905,-24.922,0.10512,0.04274,0.03143,-7.042e-06])}
df = pd.DataFrame([sun,mercury,venus,earth,mars,jupiter,saturn,uranus,neptune])
df[['coords','mass']]

# %% Helper methods for lyapunov exponant tracing graphs
FUDGE_FACTOR = 0.001
def sample_spherical(npoints,ndim=8):
    vec = np.random.randn(ndim,npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec
def get_standard_orth_basis(dim=24):
    basis = []
    for i in range(dim):
        u = np.zeros(dim)
        u[i] = 1
        u = to_supervec(u)
        basis.append(u)
    return basis

def to_supervec(u):
    if len(u)%6==0:
        return np.array([u[6*i:6*(i+1)] for i in range(int(len(u)/6))])
    else:
        raise Exception("did not give a valid array like length, must be divisible by 6, rem this is n body problem")

# normalizes the vector u, makes it have magnitude FUDGE_FACTOR
def normalize(u):
    norm_u = math.sqrt(sum([i**2 for i in u.flatten()]))
    return u/norm_u * FUDGE_FACTOR

# returns the norm \ magnitude of u
def norm(u):
    return math.sqrt(sum([i**2 for i in u.flatten()]))

def calculate_angle(u1,u2):
    # assumes u1,u2 are flat np arrays
    norm1,norm2 = np.linalg.norm(u1),np.linalg.norm(u2)
    try:
        return math.acos(np.dot(u1/norm1,u2/norm2))*180/math.pi
    except:
        return None

def rand_pert(dim=24):
    return to_supervec(sample_spherical(1,dim).T[0])
def time_evolve_randomly(Y0,steps=30,dim=24):
    # uses global variables masses,names
    solar_x = System(Y0,masses,names)
    for i in range(steps):
        u = rand_pert(dim)*random.uniform(0.1,0.2)
        solar_x.perturb(u)
        solar_x.naive_projection(first_integrals=[(System.get_energy,System.get_nabla_H),
                                                  (System.get_ang_mom_x,System.nabla_l_x),
                                                  (System.get_ang_mom_y,System.nabla_l_y),
                                                  (System.get_ang_mom_z,System.nabla_l_z),
                                                  (System.get_lin_mom_x,System.nabla_p_x),
                                                  (System.get_lin_mom_y,System.nabla_p_y),
                                                  (System.get_lin_mom_z,System.nabla_p_z)])
        solar_x.update_planets()
    return solar_x.Y


def get_lyap(Y0,steps = 150,integrator="stromer_verlet_timestep",projection=None,saveas=None,
            names = ['Sun','Jupiter','Saturn','Uranus'] ):
    # get basis of pertubations in tangent space
    FUDGE_FACTOR=0.01
    pert_basis = get_standard_orth_basis(dim=6*len(names))

    # initialize a solar system
    solar = System(Y0,masses,names=names) 

    lyap_trace = []
    for i in range(steps):
        if i%300==-1: print(i)
        # initialize 24 systems (pertubations)
        pertubations = [System(solar.Y + u * FUDGE_FACTOR,
                               masses,names=names) for u in pert_basis]
        # time evolve the system and all the pertubations one step
        solar.stromer_verlet_timestep()
        solar.update_planets() 
        for pert in pertubations:
            exec("pert.{}()".format(integrator))
            if projection: exec("pert.{}()".format(projection))
            pert.update_planets()
            
        
        # compute the jacobian and its eigenvalues
        jacobian = np.array([(pert.Y - solar.Y).flatten() / FUDGE_FACTOR for pert in pertubations])
        eigs = [i.real for i in np.linalg.eigvals(jacobian)]
        lyap = [math.log(abs(i)) for i in eigs]
        lyap.sort(reverse=True)
        lyap_trace.append(lyap)
    solar.make_trace()

    lyap_trace = np.array(lyap_trace).T
    lyap_exp = [np.mean(i) for i in lyap_trace]
    
    # display the trace and the lyap vectors
    plt.figure(figsize=(12,8))
    for exp,trace in zip(lyap_exp,lyap_trace):
        plt.plot(trace,label="{}".format(round(exp,3)))
    plt.title("Lyapunov Exponants, H={}\n sum lyap : {}".format(System.H,sum(lyap_exp)),fontsize=22)
    plt.legend()
    if saveas: plt.savefig(saveas)
    plt.show()
    
    # pot 3d trajectory of plantes
    plot3d_system(solar,k=3)
    
    return lyap_exp,lyap_trace

# 3D plots
# plots a 3d model of solar - an object of the class System
def plot3d_system(solar,k=5):
    plotly.offline.init_notebook_mode()
    data=[]
    for planet in solar.planets:

        x,y,z = np.array(planet.trace).T[3:]
        # reduce each of these
        x = [j for i,j in enumerate(x) if i%k==0]
        y = [j for i,j in enumerate(y) if i%k==0]
        z = [j for i,j in enumerate(z) if i%k==0]

        trace = go.Scatter3d(x=x,y=y,z=z,mode='markers',
                             marker={'size':1,'opacity':1.0})

        data.append(trace)

    layout = go.Layout(margin={'l':0,'r':0,'b':0,'t':0})
    plot_figure = go.Figure(data=data,layout=layout)

    plotly.offline.iplot(plot_figure)
    return




