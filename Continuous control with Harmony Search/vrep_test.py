from pyrep import PyRep
import numpy as np
import time
import random

def getSimulatorState(pyrep):
	try:
	    _, s, _, _ = pyrep.script_call(function_name_at_script_name="getState@control_script",
	                                    script_handle_or_type=1,
	                                    ints=(), floats=(), strings=(), bytes="")
	except:
	    print("Couldn't get state from VRep")
	    s = [0.0] * 7
	state = {}
	state["ball_pos"] = s[0:3]
	state["ball_vel"] = s[3:6]
	state["goal"]     = s[6]
	state["hit"]      = s[7]
	return state
 
def setupEnvironment(pyrep):
	offset =0 #np.random.uniform(-0.1, 0.1, size=None)
	_, _, _, _ = pyrep.script_call(function_name_at_script_name="setupEnvironment@control_script",
	                                script_handle_or_type=1,
	                                ints=(), floats=[offset], strings=(), bytes="")
	pyrep.step()
	return getSimulatorState(pyrep)

def kickBall(pyrep, vx, vy):
	_, _, _, _ = pyrep.script_call(function_name_at_script_name="applyKick@control_script",
	                                script_handle_or_type=1,
	                                ints=(), floats=[vx, vy], strings=(), bytes="")

def stepEnvironment(pyrep):
	state = getSimulatorState(pyrep)
	# Save the state or somehow use it for training

	pyrep.step()
	return state

def harmonysearch(pop):
	fit_val=[]
	for i in pop:
		fit_val.append(fitness(i))
		print(str(i)+"eval done")
	for k in range(improvisation):
		harmony=(random.randint(-27, 27))
		r1=random.uniform(0,1)
		if(r1<hmcr):
			harmony=harmony+1.01
			r2=random.uniform(0,1)
			if(r2<par):
				harmony=harmony + random.uniform(0,1)*bw
			if(harmony<lb):
				harmony=lb
			if(harmony>ub):
				harmony=ub
		else:
			harmony=harmony+(lb+random.uniform(0,1)*(ub-lb))/100
		new=fitness(harmony)
		if(new>min(fit_val)):
			index=fit_val.index(min(fit_val))
			fit_val[index]=new
			pop[index]=harmony
		print(k)
	print(pop)
	return pop[fit_val.index(max(fit_val))]
			
		
		
			

def goal(state):
	if state["goal"] == 1:
	    return True
	return False

def fitness(vx):
	pyrep.start()
	for i in range(max_steps):
		if(i==0):
			state=setupEnvironment(pyrep)
		if(i==1):
			#vx,vy= haveModelPredictValues()
			kickBall(pyrep,vx,vy)
		state=stepEnvironment(pyrep)
		if(goal(state)):
			fit=1+(1/i)
			break
		if(i==max_steps-1):
			fit=-1+(1/i)
	pyrep.stop()
	return fit

pyrep = PyRep()
pyrep.launch("vrep_scene_1.ttt", headless=False)
improvisation = 50
hmcr = 0.8
par = 0.25
bw=-1
hms=10
lb=-22
ub=22

max_steps = 1000
vy=-75
harmonies = []  
for j in range(hms): 
    harmonies.append(random.randint(-27, 27)) 

ans=harmonysearch(harmonies)
print("--------------Testing after optimizaton done----------------------")
f=fitness(ans)
pyrep.shutdown()
print(ans)




