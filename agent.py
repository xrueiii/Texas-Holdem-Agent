import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
import joblib
import os
from game.players import BasePokerPlayer

class GBDTAgent(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.rf_model = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=0)
        self.gbdt_model = GradientBoostingClassifier(n_estimators=80, max_depth=5, random_state=0)
        self.training_data = []  # For storing training data
        self.training_data_file = 'training_data.pkl'
        self.load_training_data()  # Load training data from file
        self.load_model()
        self.train_model()

    def load_model(self):
        try:
            with open('model.pkl', 'rb') as f:
                self.model = joblib.load(f)
            print("Trained model loaded successfully.")
        except Exception as e:
            print(f"Error loading trained model: {e}")

    def save_training_data(self):
        try:
            with open(self.training_data_file, 'wb') as f:
                joblib.dump(self.training_data, f)
            print("Training data saved successfully.")
        except Exception as e:
            print(f"Error saving training data: {e}")

    def load_training_data(self):
        if os.path.exists('training_data.pkl'):
            with open(self.training_data_file, 'rb') as f:
                self.training_data = joblib.load(f)
            print("Training data loaded.")
        else:
            print("No training data found, starting fresh.")

    def train_model(self):
        if len(self.training_data) > 1:
            features, labels = zip(*self.training_data)
            features = np.array(features)
            labels = np.array(labels)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

            unique_labels = np.unique(labels)
            
            if len(unique_labels) >= 2:
                try:
                    # Attempt to fit Gradient Boosting model
                    self.gbdt_model.fit(X_train, y_train)
                    self.model = self.gbdt_model
                    print("Using Gradient Boosting model.")
                except ValueError:
                    # Fall back to Random Forest if GBDT cannot fit
                    self.rf_model.fit(X_train, y_train)
                    self.model = self.rf_model
                    print("Using Random Forest model.")
            else:
                # Fall back to Random Forest if not enough unique classes
                self.rf_model.fit(X_train, y_train)
                self.model = self.rf_model
                print("Using Random Forest model.")

            # Save the trained model
            self.save_model()

            # Optionally, you can evaluate performance on the test set
            test_score = self.model.score(X_test, y_test)
            print(f"Test score: {test_score}")

            # Optionally, you can evaluate performance on training data itself
            try:
                train_score = self.model.score(X_train, y_train)
                print(f"Training score: {train_score}")
            except NotFittedError:
                print("Model is not fitted yet.")
                
    def save_model(self):
        try:
            with open('model.pkl', 'wb') as f:
                joblib.dump(self.model, f)
            print("Trained model saved successfully.")
        except Exception as e:
            print(f"Error saving trained model: {e}")

    def evaluate_hand_strength(self, hole_card, community_card):
        # Basic evaluation based on card ranks and suits
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suits = {'C': 1, 'D': 2, 'H': 3, 'S': 4}

        # Extract ranks and suits
        hole_ranks = [rank_values[card[1]] for card in hole_card]
        hole_suits = [suits[card[0]] for card in hole_card]

        community_ranks = [rank_values[card[1]] for card in community_card]
        community_suits = [suits[card[0]] for card in community_card]

        all_ranks = hole_ranks + community_ranks
        all_suits = hole_suits + community_suits

        # Count occurrences of ranks and suits
        def count(items):
            counts = {}
            for item in items:
                counts[item] = counts.get(item, 0) + 1
            return counts

        all_ranks_counts = count(all_ranks)
        all_suits_counts = count(all_suits)

        # Hand strength evaluation
        hand_strength = max(hole_ranks)
        
        # Check for pairs, three of a kind, four of a kind
        pairs = [rank for rank, count in all_ranks_counts.items() if count == 2]
        triples = [rank for rank, count in all_ranks_counts.items() if count == 3]
        quads = [rank for rank, count in all_ranks_counts.items() if count == 4]

        if quads:
            hand_strength += 80  # Four of a Kind
        elif triples and pairs:
            hand_strength += 70  # Full House
        elif max(all_suits_counts.values()) >= 5:
            hand_strength += 60  # Flush
        elif any(rank in all_ranks_counts and all_ranks_counts[rank] > 0 and rank + 1 in all_ranks_counts and all_ranks_counts[rank + 1] > 0 and rank + 2 in all_ranks_counts and all_ranks_counts[rank + 2] > 0 and rank + 3 in all_ranks_counts and all_ranks_counts[rank + 3] > 0 and rank + 4 in all_ranks_counts and all_ranks_counts[rank + 4] > 0 for rank in range(10)):
            hand_strength += 50  # Straight
        elif triples:
            hand_strength += 40  # Three of a Kind
        elif len(pairs) >= 2:
            hand_strength += 30  # Two Pair
        elif pairs:
            hand_strength += 20  # One Pair
        elif max(all_suits_counts.values()) >= 4:
            hand_strength += 20  # May have Flush
        elif max(all_suits_counts.values()) >= 3:
            hand_strength += 20  # May have Flush
        elif max(all_suits_counts.values()) >= 2:
            hand_strength += 10  # May have Flush
        elif (hole_ranks[0] - hole_ranks[1] == 1) or (hole_ranks[1] - hole_ranks[0] == 1):
            hand_strength += 10  # May have Straight
        
        return hand_strength
    
    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        hand_strength = self.evaluate_hand_strength(hole_card, community_card)      

        raise_amount = [action['amount']['min'] for action in valid_actions if action['action'] == 'raise']
        call_amount = [action['amount'] for action in valid_actions if action['action'] == 'call']
        raise_amount = raise_amount[0] if raise_amount else 0
        call_amount = call_amount[0] if call_amount else 0

        opponent_behavior = self.extract_opponent_behavior(round_state)
        self_money = round_state['seats'][round_state['next_player']]['stack']

        features = [hand_strength, len(community_card), raise_amount, call_amount, opponent_behavior, self_money]

        # Add logic for all-in conditions
        if self_money < 550 and hand_strength > 61:
            return 'raise', self_money  # All-in
        elif self_money < 400 and hand_strength > 51:
            return 'raise', self_money  # All-in        
        elif self_money < 300 and hand_strength > 31:
            return 'raise', self_money  # All-in
        elif self_money < 200 and hand_strength > 21:
            return 'raise', self_money  # All-in

        # Adjust raise_amount based on hand_strength
        if hand_strength > 70:
            if raise_amount < 30:
                raise_amount = raise_amount * 20  # Example: Double the raise amount
            else:
                raise_amount = raise_amount * 10
        elif hand_strength > 60:
            if raise_amount < 30:
                raise_amount = raise_amount * 15  # Example: Increase raise amount by 50%
            else:
                raise_amount = raise_amount * 10
        elif hand_strength > 50:
            if raise_amount < 30:
                raise_amount = raise_amount * 10  # Example: Increase raise amount by 20%
        else:
            raise_amount = raise_amount * 5

        # Ensure raise_amount does not exceed the maximum possible amount
        max_raise_amount = [action['amount']['max'] for action in valid_actions if action['action'] == 'raise']
        max_raise_amount = max_raise_amount[0] if max_raise_amount else self_money
        raise_amount = min(raise_amount, max_raise_amount)

        # Double the raise amount in the last two rounds
        if round_state['round_count'] > 18:
            raise_amount *= 2

        # Ensure raise_amount is not negative
        raise_amount = max(raise_amount, 0)
        
        # Consider additional heuristics to prevent folding too easily
        if call_amount > 300:
            if hand_strength > 60:
                action_pred = self.model.predict([features])[0]
            else:
                action_pred = 0
        elif call_amount > 250:
            if hand_strength > 50:
                action_pred = self.model.predict([features])[0]
            else:
                action_pred = 0
        elif call_amount > 200:
            if hand_strength > 40:
                action_pred = self.model.predict([features])[0]
            else:
                action_pred = 0
        elif call_amount > 140 and len(community_card) > 3:
            if hand_strength > 40:
                action_pred = self.model.predict([features])[0]
            else:
                action_pred = 0
        elif call_amount > 100 and len(community_card) > 3:
            if hand_strength > 30:
                action_pred = self.model.predict([features])[0]
            else:
                action_pred = 0    
        elif call_amount > 100 and len(community_card) <= 3:
            if hand_strength > 21:
                action_pred = self.model.predict([features])[0]
            else:
                action_pred = 0
        elif call_amount > 40:
            if hand_strength >= 20:
                action_pred = self.model.predict([features])[0]
            else:
                action_pred = 0
        elif hand_strength > 40:
            action_pred = 2
        elif hand_strength > 10 and len(community_card) == 0:  # Basic condition to avoid folding too easily
            action_pred = 1  # Call more often if hand strength is high or NO community card
        elif hand_strength > 9 or len(community_card) == 0:
            action_pred = self.model.predict([features])[0]
        else:
            action_pred = 0
            
        action_map = {0: 'fold', 1: 'call', 2: 'raise'}
        action = action_map.get(action_pred, 'fold')
        amount = 0

        if action == 'raise':
            amount = raise_amount
        elif action == 'call':
            amount = call_amount

        self.training_data.append((features, action_pred))  # Collect training data
        self.save_training_data()  # Save training data after each action

        return action, amount

    def extract_opponent_behavior(self, round_state):
        opponent_actions = round_state['action_histories']
        opponent_behavior = 0
        
        if opponent_actions:
            for street, actions in opponent_actions.items():
                for index, action in enumerate(actions):
                    if action['uuid'] != self.uuid:  # Ensure it's not your own action
                        if action['action'] == 'raise':
                            opponent_behavior += 1 
                        elif action['action'] == 'call':
                            opponent_behavior += 0.5 
                        elif action['action'] == 'fold':
                            opponent_behavior -= 0.5 

        return opponent_behavior


    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.train_model()  # Retrain the model after each round

def setup_ai():
    return GBDTAgent()
