import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import pickle
import hashlib
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Experience:
    """Experience replay buffer entry"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    metadata: Dict[str, Any]

class DeepfakeDetectionEnvironment:
    """Environment for deepfake detection RL agent"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.action_space = 3  # [classify_real, classify_fake, uncertain]
        self.observation_space = 512  # Feature vector size
        self.current_state = None
        self.episode_reward = 0
        self.episode_steps = 0
        self.max_steps = 100
        
    def reset(self, features: np.ndarray) -> np.ndarray:
        """Reset environment with new image features"""
        self.current_state = features
        self.episode_reward = 0
        self.episode_steps = 0
        return self.current_state
    
    def step(self, action: int, ground_truth: int, confidence: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and return next state, reward, done, info"""
        self.episode_steps += 1
        
        # Calculate reward based on action correctness and confidence
        reward = self._calculate_reward(action, ground_truth, confidence)
        self.episode_reward += reward
        
        # Determine if episode is done
        done = self.episode_steps >= self.max_steps
        
        # Next state remains the same for single image classification
        next_state = self.current_state
        
        info = {
            "action": action,
            "ground_truth": ground_truth,
            "confidence": confidence,
            "reward": reward,
            "episode_reward": self.episode_reward
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action: int, ground_truth: int, confidence: float) -> float:
        """Calculate reward based on action correctness and confidence"""
        # Base reward for correct classification
        if action == ground_truth:
            base_reward = 1.0
        else:
            base_reward = -1.0
        
        # Confidence bonus/penalty
        confidence_bonus = confidence if action == ground_truth else (1 - confidence)
        
        # Uncertainty penalty for high confidence wrong predictions
        if action != ground_truth and confidence > 0.8:
            confidence_bonus -= 0.5
        
        return base_reward + confidence_bonus

class PolicyNetwork(nn.Module):
    """Policy network for the RL agent"""
    
    def __init__(self, input_size: int = 512, hidden_size: int = 256, num_actions: int = 3):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    """Value network for the RL agent"""
    
    def __init__(self, input_size: int = 512, hidden_size: int = 256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ExperienceReplayBuffer:
    """Experience replay buffer for storing and sampling experiences"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def add(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return np.random.choice(list(self.buffer), batch_size, replace=False).tolist()
    
    def __len__(self) -> int:
        return len(self.buffer)

class DeepfakeRLAgent:
    """Reinforcement Learning agent for deepfake detection"""
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 buffer_size: int = 10000,
                 batch_size: int = 32):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        
        # Initialize networks
        self.policy_net = PolicyNetwork().to(self.device)
        self.value_net = ValueNetwork().to(self.device)
        self.target_policy_net = PolicyNetwork().to(self.device)
        self.target_value_net = ValueNetwork().to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer and environment
        self.replay_buffer = ExperienceReplayBuffer(buffer_size)
        self.environment = DeepfakeDetectionEnvironment()
        
        # Training statistics
        self.training_stats = {
            "episodes": 0,
            "total_reward": 0,
            "avg_reward": 0,
            "accuracy": 0,
            "losses": []
        }
        
        # Load existing model if available
        self._load_model()
        
    def _load_model(self):
        """Load pre-trained model if available"""
        model_path = Path("models/rl_agent.pth")
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.value_net.load_state_dict(checkpoint['value_net'])
                self.training_stats = checkpoint.get('training_stats', self.training_stats)
                logger.info("Loaded pre-trained RL model")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
    
    def _save_model(self):
        """Save current model"""
        model_path = Path("models")
        model_path.mkdir(exist_ok=True)
        
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'training_stats': self.training_stats
        }
        torch.save(checkpoint, model_path / "rl_agent.pth")
        logger.info("Saved RL model")
    
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """Get action from current state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            action_values = self.value_net(state_tensor)
        
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            action = np.random.randint(0, 3)
        else:
            # Exploitation: best action
            action = torch.argmax(action_probs).item()
        
        confidence = action_probs[0][action].item()
        return action, confidence
    
    def update_target_networks(self):
        """Update target networks with current networks"""
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_value_net.load_state_dict(self.value_net.state_dict())
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return {"policy_loss": 0, "value_loss": 0}
        
        # Sample batch from replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        
        # Compute current Q values
        current_action_probs = self.policy_net(states)
        current_values = self.value_net(states)
        
        # Compute target Q values
        with torch.no_grad():
            next_action_probs = self.target_policy_net(next_states)
            next_values = self.target_value_net(next_states)
            target_values = rewards + (self.gamma * next_values.squeeze() * ~dones)
        
        # Policy loss (cross-entropy with advantage)
        advantage = (target_values - current_values.squeeze()).detach()
        policy_loss = F.cross_entropy(current_action_probs, actions, reduction='none')
        policy_loss = (policy_loss * advantage).mean()
        
        # Value loss (MSE)
        value_loss = F.mse_loss(current_values.squeeze(), target_values)
        
        # Update networks
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }
    
    def learn_from_feedback(self, 
                           features: np.ndarray,
                           predicted_action: int,
                           ground_truth: int,
                           confidence: float,
                           user_feedback: Optional[bool] = None) -> Dict[str, Any]:
        """Learn from user feedback and ground truth"""
        
        # Reset environment with new features
        state = self.environment.reset(features)
        
        # Get action from current policy
        action, pred_confidence = self.get_action(state, training=False)
        
        # Use user feedback if available, otherwise use ground truth
        if user_feedback is not None:
            reward = 1.0 if user_feedback else -1.0
        else:
            # Calculate reward based on ground truth
            reward = self.environment._calculate_reward(predicted_action, ground_truth, confidence)
        
        # Create experience
        experience = Experience(
            state=state,
            action=predicted_action,
            reward=reward,
            next_state=state,  # Same state for single image classification
            done=True,
            metadata={
                "ground_truth": ground_truth,
                "confidence": confidence,
                "user_feedback": user_feedback,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Add to replay buffer
        self.replay_buffer.add(experience)
        
        # Train the agent
        losses = self.train_step()
        
        # Update training statistics
        self.training_stats["episodes"] += 1
        self.training_stats["total_reward"] += reward
        
        # Update target networks periodically
        if self.training_stats["episodes"] % 100 == 0:
            self.update_target_networks()
        
        # Save model periodically
        if self.training_stats["episodes"] % 1000 == 0:
            self._save_model()
        
        return {
            "action": predicted_action,
            "confidence": pred_confidence,
            "reward": reward,
            "losses": losses,
            "training_stats": self.training_stats
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics"""
        stats = self.training_stats.copy()
        if stats["episodes"] > 0:
            stats["avg_reward"] = stats["total_reward"] / stats["episodes"]
        return stats
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        if len(self.replay_buffer) == 0:
            return {"accuracy": 0, "avg_confidence": 0, "total_experiences": 0}
        
        # Calculate accuracy from recent experiences
        recent_experiences = list(self.replay_buffer)[-100:]
        correct_predictions = sum(1 for exp in recent_experiences 
                                if exp.metadata.get("ground_truth") == exp.action)
        accuracy = correct_predictions / len(recent_experiences)
        
        avg_confidence = np.mean([exp.metadata.get("confidence", 0) 
                                for exp in recent_experiences])
        
        return {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "total_experiences": len(self.replay_buffer),
            "recent_experiences": len(recent_experiences)
        }

class ReinforcementLearningService:
    """Service for managing reinforcement learning in the deepfake detection system"""
    
    def __init__(self):
        self.agent = DeepfakeRLAgent()
        self.feature_extractor = None  # Will be set by the main detector
        self.learning_enabled = True
        
    def set_feature_extractor(self, feature_extractor):
        """Set the feature extractor from the main detector"""
        self.feature_extractor = feature_extractor
    
    def process_user_feedback(self, 
                            file_id: str,
                            features: np.ndarray,
                            prediction: Dict[str, Any],
                            user_feedback: bool) -> Dict[str, Any]:
        """Process user feedback and update the RL agent"""
        
        if not self.learning_enabled:
            return {"message": "Learning disabled"}
        
        try:
            # Extract prediction details
            predicted_class = prediction.get("class", 0)
            confidence = prediction.get("confidence", 0.5)
            
            # Learn from feedback
            result = self.agent.learn_from_feedback(
                features=features,
                predicted_action=predicted_class,
                ground_truth=1 if user_feedback else 0,  # Assuming binary classification
                confidence=confidence,
                user_feedback=user_feedback
            )
            
            # Store feedback for analysis
            self._store_feedback(file_id, prediction, user_feedback, result)
            
            return {
                "status": "success",
                "learning_result": result,
                "message": "Feedback processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error processing user feedback: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning process"""
        training_stats = self.agent.get_training_stats()
        performance = self.agent.get_model_performance()
        
        return {
            "training_stats": training_stats,
            "performance": performance,
            "learning_enabled": self.learning_enabled,
            "model_info": {
                "policy_network_params": sum(p.numel() for p in self.agent.policy_net.parameters()),
                "value_network_params": sum(p.numel() for p in self.agent.value_net.parameters()),
                "device": str(self.agent.device)
            }
        }
    
    def _store_feedback(self, file_id: str, prediction: Dict, user_feedback: bool, learning_result: Dict):
        """Store feedback data for analysis"""
        feedback_data = {
            "file_id": file_id,
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "user_feedback": user_feedback,
            "learning_result": learning_result
        }
        
        # Store in feedback database
        feedback_path = Path("data/feedback")
        feedback_path.mkdir(parents=True, exist_ok=True)
        
        feedback_file = feedback_path / f"feedback_{file_id}.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
    
    def enable_learning(self, enabled: bool = True):
        """Enable or disable learning"""
        self.learning_enabled = enabled
        logger.info(f"Learning {'enabled' if enabled else 'disabled'}")
    
    def reset_learning(self):
        """Reset the learning agent"""
        self.agent = DeepfakeRLAgent()
        logger.info("Learning agent reset")
    
    def export_learning_data(self) -> Dict[str, Any]:
        """Export learning data for analysis"""
        return {
            "training_stats": self.agent.get_training_stats(),
            "performance": self.agent.get_model_performance(),
            "replay_buffer_size": len(self.agent.replay_buffer),
            "model_state": {
                "policy_net": self.agent.policy_net.state_dict(),
                "value_net": self.agent.value_net.state_dict()
            }
        } 