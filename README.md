# SHERLOCK : Twitch-Chatbot-detection (Livestreaming platform)

## Objective
The purpose of our project is to detect chatbots in a chat room of a stream at a real time to deduce fraudulent activities.

## Problem
We formulate the problem into two subproblems. First is to detect whether the stream is chat botted in real time. Secondly if it is then find out the chat bots in that corrupted stream.

## Solution
* For Stage 1, we have used supervised approach to predict whether the given stream is chatbotted or not.
* For Stage 2, we took the help of semi supervised approach `Label Propagation` due to less number of labelled users.

## Simulation
* To detect whether the given stream is botted, execute the following command:
`python StreamClassification.py`
* To determine the constituent bot users in corrupted stream, run the following command:
`python modified_main.py <Merged Users filename> <Real Users filename>`
Here` Merged Users filename` is a corrupted file with both real and bot users while `Real Users filename` consists of corresponding legitimate real users.


## Note
* Use `python>=2.7.13`
* Write to us for any implementation clarity or dataset request at shreya.jain@research.iiit.ac.in or dipankar.niranjan@research.iiit.ac.in.
