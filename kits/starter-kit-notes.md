# Notes for future starter kits

There are a few gotchas one may run into when developing a starter kit. Most of them are relevant for statically
typed programming languages.

It is obviously recommended to use the existing kits as reference. But this list may still help to catch issues that
are not too obvious in the existing code.

- [1 Compiled kits](#1-compiled-kits)
- [2 main.py](#2-mainpy)
- [3 Input/Output](#3-inputoutput)
- [4 Configuration](#4-configuration)

## 1 Compiled kits

Kaggle servers currently run Ubuntu 18.04. Kits that require compilation to a native binary should support a build
using docker. That way everyone, no matter the OS, can submit their agent to Kaggle.
A native build should also be available for local testing.

## 2 main.py

To make the kit work on Kaggle the submission needs a `main.py` which creates a process from the binary built with
docker. Check the existing kits for reference of such a `main.py`.

## 3 Input/Output

Your kit receives input from `stdin` and communicates its output to `stdout`. The format used for communication is `JSON`.
Any form of logging (e.g. for debugging) can be directed to `stderr`.

Your agent will run as long as there is input available.
Every turn, your agent must provide output. (at least an empty `JSON` object)

## 4 Configuration

The partial game configuration under `"info"."env_cfg"` is only available on the first step. (`"step"` = 0)

## 5 Observation

See the sample_step_0_input.txt and sample_step_input.txt files in this folder for what the returned string from the game engine when parsed as JSON looks like.

## 6 Actions

The action space of the game is always a fixed `(N, 3)` array of integers to control up to units `0` to `N-1` on your team where `N` is the max number of units each team can have (example code shows how to determine `N`). At any given point in time you might not have `N` units on your team so actions for those nonexistent units do not do anything.

For each unit's action, the first integer indicates the type of action, which can be 0 for doing nothing, 1 to move up, 2 to move right, 3 to move down, 4 to move left, and 5 to sap a tile. The next 2 integers are only for the sap action and indicate the location of the tile to sap from relative to the unit's position (a delta x and y value).

To actually submit actions to the game engine you just need to print out in JSON format an object of the following form:

```js
{
    "actions": ...
}
```

where the actions value is a string representing the N, 3 matrix/array of action values.