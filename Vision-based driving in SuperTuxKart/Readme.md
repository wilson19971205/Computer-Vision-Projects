# Introduction
This is a vision-based driving agent in SuperTuxKart.

# Collecting Data
Designing a controller that acts as an auto-pilot to drive in SuperTuxKart, then use this auto-pilot to train a vision-based driving system.

Use the following code to install SuperTuxKart:
```python
%pip install -U PySuperTuxKart
```

Gathering the images that aim point on the center of the track 15 meters away from the kart, as shown below.

![image](video/aim_point.png)

Collecting data from 6 tracks. 
Zen garden, Lighthouse, Hacienda, Snowtuxpeak, Cornfield crossing and Scotland.

### Controller
These of the hyper-parameters of the controller.
pystk.Action.steer: the steering angle of the kart normalized to -1 ... 1
pystk.Action.acceleration: the acceleration of the kart normalized to 0 ... 1
pystk.Action.brake: a boolean indicator for braking
pystk.Action.drift: a special action that makes the kart drift, useful for tight turns
pystk.Action.nitro: burns nitro for fast acceleration


This is what it looks like:

https://github.com/wilson19971205/Computer-Vision-Projects/assets/43212302/fb92092b-e3c2-4bdb-b698-1fe2ce08975f

https://github.com/wilson19971205/Computer-Vision-Projects/assets/43212302/d876e968-5cb6-4c93-9bfc-27e401d8b686

# Vision-Based Driving
The <span style="color: green"> red </span> circle in the video below is the true aim point on the center of the track 15 meters away from the kart.

The <span style="color: green"> green </span> circle is the predicted point.

The <span style="color: green"> blue </span> circle is the position of the kart.

The two videos below demonstrate the vision-driving under the track in train data:

https://github.com/wilson19971205/Computer-Vision-Projects/assets/43212302/31e9ca6c-42bd-4397-8bf2-47448d724dc0

https://github.com/wilson19971205/Computer-Vision-Projects/assets/43212302/fbb030aa-d4da-429e-ad31-87ca7242d3d1


This one is the track that the model hadn't seen before:

https://github.com/wilson19971205/Computer-Vision-Projects/assets/43212302/5073d040-49dd-4e7f-b1a8-37abc1eb4cea

