# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:03:10 2020

@author: user
"""

import numpy as np
import bj_brain as brain

def pack():
    pack_of_cards = np.arange(52)
    return pack_of_cards

def card_shape(num):
    if num == 0:
        return "Hearts"
    if num == 1:
        return "Clubs"
    if num == 2:
        return "Diamonds"
    if num == 3:
        return "Spades"

def draw(pack):
    random_i = np.random.randint(pack.shape[0])
    drawn_card = pack[random_i]
    drawn_value = drawn_card%13 +1

    if drawn_value > 10:
        drawn_value = 10
    pack = np.delete(pack,random_i)
    return pack, drawn_value 

#Standard game of blackjack
#Player actions depend on ANN output
#Player can DOUBLE or DRAW

def game(ann_parameters):
    #Create a standard 52 cards pack
    pack_of_cards = pack()
    #Player can draw up to 7 cards in total
    player_cards = np.zeros(7)
    #Dealer starts with 2 cards and has to draw until it reaches 17
    dealer_cards = np.zeros(2)
    
    #Setting basic bet
    bet = 1
    
    #Drawing 2 cards for each side
    for turn in range(2):
        pack_of_cards, player_cards[turn] = draw(pack_of_cards)
        pack_of_cards, dealer_cards[turn] = draw(pack_of_cards)

    #Counting the number of cards in the player's hand
    player_count = 2
    
    #Player ANN can see its own cards and the dealer's first card
    #Decision to DOUBLE:
    if (brain.ann(ann_parameters, player_cards, dealer_cards[0])[0] >= 0):
        #DOUBLE THE BET!
        bet = 2
        #Player has to draw once
        pack_of_cards, player_cards[player_count] = draw(pack_of_cards)
        player_count += 1
    else:
        #Decision to DRAW:
        while (brain.ann(ann_parameters, player_cards, dealer_cards[0])[1] >= 0) and (player_count < 7) and (np.sum(player_cards)<21):
            #DRAW A CARD!
            pack_of_cards, player_cards[player_count] = draw(pack_of_cards)
            player_count += 1
    
    if (np.sum(player_cards) > 21):
        #LOSE!
        return (-2*bet)
    else:
        #Converting ACES from 1 to 11 if it benefits the dealer
        dealer_sum = np.sum(dealer_cards)
        if (np.sum(np.where(dealer_cards==1, 1,0)) > 0) and (6 < np.sum(dealer_cards) < 12):
            dealer_sum += 10
    
        #Whether the dealer hasn't reach 17, it has to draw until it does
        while (dealer_sum < 17):
            dealer_cards = np.append(dealer_cards,0)
            pack_of_cards, dealer_cards[dealer_cards.shape[0]-1] = draw(pack_of_cards)
            if (np.sum(np.where(dealer_cards==1, 1,0)) > 0) and (6 < np.sum(dealer_cards) < 12):
                dealer_sum = np.sum(dealer_cards) + 10
            else: 
                dealer_sum = np.sum(dealer_cards) 
    
        #Converting ACES from 1 to 11 if it benefits the player
        player_sum = np.sum(player_cards)
        if (np.sum(np.where(player_cards==1, 1,0)) > 0) and (player_sum < 12):
            player_sum += 10
    
        #Outcomes:
        if player_sum<21:
            if player_sum<dealer_sum<=21:
                #LOSE!
                return (-2*bet)
            elif dealer_sum<player_sum or 21<dealer_sum:
                #WIN!
                return (2*bet)
            else:
                #PUSH!
                return (bet)
        elif player_sum==21:
                if player_sum==dealer_sum:
                    #PUSH!
                    return (bet)
                else:
                    #BLACKJACK!
                    return (2.5*bet)
    
    
        
        

def game_viz(ann_parameters):
    #Create a standard 52 cards pack
    pack_of_cards = pack()
    #Player can draw up to 7 cards in total
    player_cards = np.zeros(7)
    #Dealer starts with 2 cards and has to draw until it reaches 17
    dealer_cards = np.zeros(2)
    
    #Setting basic bet
    bet = 1
    
    #Drawing 2 cards for each side
    for turn in range(2):
        pack_of_cards, player_cards[turn] = draw(pack_of_cards)
        pack_of_cards, dealer_cards[turn] = draw(pack_of_cards)

    #Counting the number of cards in the player's hand
    player_count = 2
    
    #Show cards:
    print("PLAYER CARDS:", player_cards)
    print("DEALER CARDS:", dealer_cards)
    
    #Player ANN can see its own cards and the dealer's first card
    #Decision to DOUBLE:
    if (brain.ann(ann_parameters, player_cards, dealer_cards[0])[0] >= 0):
        #DOUBLE THE BET!
        bet = 2
        print("PLAYER DOUBLES DOWN")
        #Player has to draw once
        pack_of_cards, player_cards[player_count] = draw(pack_of_cards)
        player_count += 1
        print("PLAYER HAS TO DRAW A CARD:", player_cards)
    else:
        #Decision to DRAW:
        while (brain.ann(ann_parameters, player_cards, dealer_cards[0])[1] >= 0) and (player_count < 7) and (np.sum(player_cards)<21):
            #DRAW A CARD!
            pack_of_cards, player_cards[player_count] = draw(pack_of_cards)
            player_count += 1
            print("PLAYER HITS:", player_cards)
    
    if (np.sum(player_cards) > 21):
        #LOSE!
        print("PLAYER LOSES")
        return (-2*bet)
    else:
        #Converting ACES from 1 to 11 if it benefits the dealer
        dealer_sum = np.sum(dealer_cards)
        if (np.sum(np.where(dealer_cards==1, 1,0)) > 0) and (6 < np.sum(dealer_cards) < 12):
            dealer_sum += 10
    
        #Whether the dealer hasn't reach 17, it has to draw until it does
        while (dealer_sum < 17):
            dealer_cards = np.append(dealer_cards,0)
            pack_of_cards, dealer_cards[dealer_cards.shape[0]-1] = draw(pack_of_cards)
            print("DEALER DREW A CARD:", dealer_cards)
            if (np.sum(np.where(dealer_cards==1, 1,0)) > 0) and (6 < np.sum(dealer_cards) < 12):
                dealer_sum = np.sum(dealer_cards) + 10
            else: 
                dealer_sum = np.sum(dealer_cards) 
    
        #Converting ACES from 1 to 11 if it benefits the player
        player_sum = np.sum(player_cards)
        if (np.sum(np.where(player_cards==1, 1,0)) > 0) and (player_sum < 12):
            player_sum += 10
    
        #Outcomes:
        if player_sum<21:
            if player_sum<dealer_sum<=21:
                #LOSE!
                print("PLAYER LOSES")
                return (-2*bet)
            elif dealer_sum<player_sum or 21<dealer_sum:
                #WIN!
                print("PLAYER WINS!")
                return (2*bet)
            else:
                #PUSH!
                print("PUSH")
                return (bet)
        elif player_sum==21:
                if player_sum==dealer_sum:
                    #PUSH!
                    print("PUSH")
                    return (bet)
                else:
                    #BLACKJACK!
                    print("PLAYER WINS BLACKJACK!")
                    return (2.5*bet)
   

def win_mean(parameters,number_of_games):
    winnings = 0
    for i in range(number_of_games):
        winnings += game(parameters)
    return (winnings/number_of_games) 

def win_mean_viz(parameters,number_of_games):
    winnings = 0
    for i in range(number_of_games):
        winnings += game_viz(parameters)
    return (winnings/number_of_games)  