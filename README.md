# ImpactX
This repository provides a simple simulation of planetary collisions using OpenGL. For collisions, only gravity and elastic collision are considered. Please make sure that cuda is installed when you run the simulation, because it uses cuda for speed-up.

<img src="planetary_impact.gif" width='600'>

<br></br>

# Elastic collision
When particles collide with each other, it is necessary to calculate the velocity after the collision according to the idea of elastic collision. Comprehensive information about elastic collision can be found [here](https://en.wikipedia.org/wiki/Elastic_collision#CITEREFSerwayJewett2014).

Consider particles A and B with masses $m_A$, $m_B$, and velocities $v_{A1}$, $v_{B1}$ before collision, $v_{A2}$, $v_{B2}$ after collision. The conservation of momentum before and after the collision is expressed by:

$$
m_A v_{A1} + m_B v_{B1} = m_A v_{A2} + m_B v_{B2} \tag{1}
$$

Likewise, the conservation of the total kenetic energey is expressed by:

$$
\frac{1}{2} m_A v_{A1}^2 + \frac{1}{2} m_B v_{B1}^2 = \frac{1}{2} m_A v_{A2}^2 + \frac{1}{2} m_A v_{B2}^2
$$

These equations may be solved directly to find $v_{A2}$, $v_{B2}$ when $v_{A1}$, $v_{B1}$ are known.

$$
\begin{align*}
v_{A2} &= \frac{m_A - m_B}{m_A + m_B} v_{A1} + \frac{2m_B}{m_A + m_B} v_{B1} \\
v_{B2} &= \frac{2m_A}{m_A + m_B} v_{A1} + \frac{m_B - m_A}{m_A + m_B} v_{B1}
\end{align*}
$$

In this case, we need to think about 3D collision with two moving particles.

In an angle-free representation, the changed velocities are computed using the center $x_1$ and $x_2$ at the time of contact as

$$
v_1 \prime = v_1 - \frac{2m_2}{m_1 + m_2} \frac{\langle v_1 - v_2, x_1 - x_2 \rangle}{\parallel x_1 - x_2 \parallel^2} (x_1 - x_2), \\
v_2 \prime = v_2 - \frac{2m_1}{m_1 + m_2} \frac{\langle v_2 - v_1, x_2 - x_1 \rangle}{\parallel x_2 - x_1 \parallel^2} (x_2 - x_1)
$$

where the angle brackets indicate the inner product of two vectors.


<br></br>

# Velocity update

<br></br>

# How to run
The tested environment is as follows.

```
- OS -> Ubuntu
- Cuda -> 11.8
- OpenGL -> 4.0
```

First, you should intall glfw on your environemt by running following command.

```bash
bash setup.sh
```

After that, you can compile and run the program with following commands.

```bash
cd srcs
make
./ImpactX
```

<br></br>

# References
- [Elastic collision](https://en.wikipedia.org/wiki/Elastic_collision#CITEREFSerwayJewett2014)
- [Elastic Collisions](https://williamecraver.wixsite.com/elastic-equations)
- [The Barnes-Hut Algorithm](http://arborjs.org/docs/barnes-hut)
