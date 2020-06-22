# RoFish

*RoFish_cpg* is an example sketch for the Arduino to demonstrate the Central Pattern Generator (CPG) locomotion control of a robotic fish. The CPG from the example *RoFish_cpg* has three coupled oscillators for driving an equal number of joints of the robotic fish. The frequency of the oscillators, the amplitude of the motion and the angle-offset, used for steering, can be set via a terminal console, connecting with the UART of the Arduino. The swimming gait can be switched between forward and backward direction.
This example has not been completely worked out with remote control. It is only meant to demonstrate the CPG. 


![CPG swimming gait](figures/animation.gif  "Animation")

The updates made to this repository include changes required to fit the CPG developed for RoFish to a quadruped spider robot.
