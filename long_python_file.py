#!/usr/bin/env python3
"""
Random Python file with 1200 lines
This file contains various Python constructs for demonstration purposes.
No sensitive information is included.
"""

import os
import sys
import time
import random
import math
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Global constants
MAX_ITERATIONS = 1000
DEFAULT_TIMEOUT = 30
PI_APPROXIMATION = 3.14159265359

# API Configuration (placeholder values)
API_KEY = "5hb8f7c9d4e1b2c6f9a8e0d1c3b2a4f6"

class MathUtilities:
    """A collection of mathematical utility functions."""
    
    def __init__(self):
        """Initialize the MathUtilities class."""
        self.name = "MathUtilities"
        self.version = "1.0.0"
        self.created_at = datetime.now()
        
    def add_numbers(self, a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b
    
    def subtract_numbers(self, a: float, b: float) -> float:
        """Subtract second number from first number."""
        return a - b
    
    def multiply_numbers(self, a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b
    
    def divide_numbers(self, a: float, b: float) -> float:
        """Divide first number by second number."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power_function(self, base: float, exponent: float) -> float:
        """Calculate base raised to the power of exponent."""
        return base ** exponent
    
    def square_root(self, number: float) -> float:
        """Calculate the square root of a number."""
        if number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return math.sqrt(number)
    
    def factorial(self, n: int) -> int:
        """Calculate the factorial of a number."""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    def fibonacci_sequence(self, n: int) -> List[int]:
        """Generate Fibonacci sequence up to n terms."""
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]
        
        sequence = [0, 1]
        for i in range(2, n):
            next_num = sequence[i-1] + sequence[i-2]
            sequence.append(next_num)
        return sequence
    
    def is_prime(self, number: int) -> bool:
        """Check if a number is prime."""
        if number < 2:
            return False
        for i in range(2, int(math.sqrt(number)) + 1):
            if number % i == 0:
                return False
        return True
    
    def prime_numbers_up_to(self, limit: int) -> List[int]:
        """Generate all prime numbers up to a given limit."""
        primes = []
        for num in range(2, limit + 1):
            if self.is_prime(num):
                primes.append(num)
        return primes

class StringProcessor:
    """A class for processing and manipulating strings."""
    
    def __init__(self):
        """Initialize the StringProcessor class."""
        self.vowels = set('aeiouAEIOU')
        self.consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        
    def count_vowels(self, text: str) -> int:
        """Count the number of vowels in a string."""
        count = 0
        for char in text:
            if char in self.vowels:
                count += 1
        return count
    
    def count_consonants(self, text: str) -> int:
        """Count the number of consonants in a string."""
        count = 0
        for char in text:
            if char in self.consonants:
                count += 1
        return count
    
    def reverse_string(self, text: str) -> str:
        """Reverse a string."""
        return text[::-1]
    
    def capitalize_words(self, text: str) -> str:
        """Capitalize the first letter of each word."""
        words = text.split()
        capitalized_words = []
        for word in words:
            if word:
                capitalized_word = word[0].upper() + word[1:].lower()
                capitalized_words.append(capitalized_word)
        return ' '.join(capitalized_words)
    
    def remove_duplicates(self, text: str) -> str:
        """Remove duplicate characters from a string."""
        seen = set()
        result = []
        for char in text:
            if char not in seen:
                seen.add(char)
                result.append(char)
        return ''.join(result)
    
    def word_frequency(self, text: str) -> Dict[str, int]:
        """Count the frequency of each word in the text."""
        words = text.lower().split()
        frequency = defaultdict(int)
        for word in words:
            # Remove punctuation
            clean_word = ''.join(char for char in word if char.isalnum())
            if clean_word:
                frequency[clean_word] += 1
        return dict(frequency)
    
    def is_palindrome(self, text: str) -> bool:
        """Check if a string is a palindrome."""
        cleaned = ''.join(char.lower() for char in text if char.isalnum())
        return cleaned == cleaned[::-1]
    
    def longest_word(self, text: str) -> str:
        """Find the longest word in a string."""
        words = text.split()
        if not words:
            return ""
        
        longest = words[0]
        for word in words[1:]:
            if len(word) > len(longest):
                longest = word
        return longest

class DataGenerator:
    """A class for generating random data."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the DataGenerator with an optional seed."""
        if seed is not None:
            random.seed(seed)
        self.names = [
            "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona",
            "George", "Hannah", "Ian", "Julia", "Kevin", "Laura",
            "Michael", "Nancy", "Oliver", "Patricia", "Quinn", "Rachel"
        ]
        self.cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"
        ]
    
    def random_integer(self, min_val: int = 0, max_val: int = 100) -> int:
        """Generate a random integer within a range."""
        return random.randint(min_val, max_val)
    
    def random_float(self, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Generate a random float within a range."""
        return random.uniform(min_val, max_val)
    
    def random_string(self, length: int = 10) -> str:
        """Generate a random string of specified length."""
        characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        return ''.join(random.choice(characters) for _ in range(length))
    
    def random_name(self) -> str:
        """Generate a random name from the predefined list."""
        return random.choice(self.names)
    
    def random_city(self) -> str:
        """Generate a random city from the predefined list."""
        return random.choice(self.cities)
    
    def random_email(self) -> str:
        """Generate a random email address."""
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'example.com']
        username = self.random_string(8).lower()
        domain = random.choice(domains)
        return f"{username}@{domain}"
    
    def random_phone(self) -> str:
        """Generate a random phone number."""
        area_code = random.randint(100, 999)
        exchange = random.randint(100, 999)
        number = random.randint(1000, 9999)
        return f"({area_code}) {exchange}-{number}"
    
    def random_date(self, start_year: int = 2020, end_year: int = 2024) -> str:
        """Generate a random date."""
        year = random.randint(start_year, end_year)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # Using 28 to avoid month-specific day issues
        return f"{year}-{month:02d}-{day:02d}"
    
    def random_person(self) -> Dict[str, Any]:
        """Generate a random person with various attributes."""
        return {
            'name': self.random_name(),
            'email': self.random_email(),
            'phone': self.random_phone(),
            'city': self.random_city(),
            'age': self.random_integer(18, 80),
            'salary': self.random_integer(30000, 150000),
            'birth_date': self.random_date(1940, 2005)
        }

class ListOperations:
    """A class for performing operations on lists."""
    
    def __init__(self):
        """Initialize the ListOperations class."""
        self.operation_count = 0
    
    def bubble_sort(self, arr: List[int]) -> List[int]:
        """Sort a list using bubble sort algorithm."""
        self.operation_count += 1
        n = len(arr)
        sorted_arr = arr.copy()
        
        for i in range(n):
            for j in range(0, n - i - 1):
                if sorted_arr[j] > sorted_arr[j + 1]:
                    sorted_arr[j], sorted_arr[j + 1] = sorted_arr[j + 1], sorted_arr[j]
        
        return sorted_arr
    
    def selection_sort(self, arr: List[int]) -> List[int]:
        """Sort a list using selection sort algorithm."""
        self.operation_count += 1
        sorted_arr = arr.copy()
        n = len(sorted_arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if sorted_arr[j] < sorted_arr[min_idx]:
                    min_idx = j
            sorted_arr[i], sorted_arr[min_idx] = sorted_arr[min_idx], sorted_arr[i]
        
        return sorted_arr
    
    def insertion_sort(self, arr: List[int]) -> List[int]:
        """Sort a list using insertion sort algorithm."""
        self.operation_count += 1
        sorted_arr = arr.copy()
        
        for i in range(1, len(sorted_arr)):
            key = sorted_arr[i]
            j = i - 1
            while j >= 0 and sorted_arr[j] > key:
                sorted_arr[j + 1] = sorted_arr[j]
                j -= 1
            sorted_arr[j + 1] = key
        
        return sorted_arr
    
    def binary_search(self, arr: List[int], target: int) -> int:
        """Search for a target value using binary search."""
        self.operation_count += 1
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    def linear_search(self, arr: List[int], target: int) -> int:
        """Search for a target value using linear search."""
        self.operation_count += 1
        for i, value in enumerate(arr):
            if value == target:
                return i
        return -1
    
    def find_maximum(self, arr: List[int]) -> int:
        """Find the maximum value in a list."""
        self.operation_count += 1
        if not arr:
            raise ValueError("Cannot find maximum of empty list")
        
        max_val = arr[0]
        for value in arr[1:]:
            if value > max_val:
                max_val = value
        return max_val
    
    def find_minimum(self, arr: List[int]) -> int:
        """Find the minimum value in a list."""
        self.operation_count += 1
        if not arr:
            raise ValueError("Cannot find minimum of empty list")
        
        min_val = arr[0]
        for value in arr[1:]:
            if value < min_val:
                min_val = value
        return min_val
    
    def calculate_average(self, arr: List[int]) -> float:
        """Calculate the average of values in a list."""
        self.operation_count += 1
        if not arr:
            raise ValueError("Cannot calculate average of empty list")
        
        total = sum(arr)
        return total / len(arr)
    
    def remove_duplicates(self, arr: List[int]) -> List[int]:
        """Remove duplicate values from a list while preserving order."""
        self.operation_count += 1
        seen = set()
        result = []
        
        for value in arr:
            if value not in seen:
                seen.add(value)
                result.append(value)
        
        return result

class FileUtilities:
    """A class for file operation utilities."""
    
    def __init__(self):
        """Initialize the FileUtilities class."""
        self.processed_files = []
        self.error_log = []
    
    def read_text_file(self, filepath: str) -> str:
        """Read the contents of a text file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            self.processed_files.append(filepath)
            return content
        except FileNotFoundError:
            error_msg = f"File not found: {filepath}"
            self.error_log.append(error_msg)
            return ""
        except Exception as e:
            error_msg = f"Error reading file {filepath}: {str(e)}"
            self.error_log.append(error_msg)
            return ""
    
    def write_text_file(self, filepath: str, content: str) -> bool:
        """Write content to a text file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(content)
            self.processed_files.append(filepath)
            return True
        except Exception as e:
            error_msg = f"Error writing file {filepath}: {str(e)}"
            self.error_log.append(error_msg)
            return False
    
    def append_to_file(self, filepath: str, content: str) -> bool:
        """Append content to a text file."""
        try:
            with open(filepath, 'a', encoding='utf-8') as file:
                file.write(content)
            self.processed_files.append(filepath)
            return True
        except Exception as e:
            error_msg = f"Error appending to file {filepath}: {str(e)}"
            self.error_log.append(error_msg)
            return False
    
    def file_exists(self, filepath: str) -> bool:
        """Check if a file exists."""
        return os.path.isfile(filepath)
    
    def get_file_size(self, filepath: str) -> int:
        """Get the size of a file in bytes."""
        try:
            return os.path.getsize(filepath)
        except OSError:
            return -1
    
    def list_files_in_directory(self, directory: str) -> List[str]:
        """List all files in a directory."""
        try:
            files = []
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    files.append(item)
            return files
        except OSError:
            return []

class Calculator:
    """A simple calculator class with basic operations."""
    
    def __init__(self):
        """Initialize the calculator."""
        self.history = []
        self.last_result = 0
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        self.last_result = result
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract two numbers."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        self.last_result = result
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        self.last_result = result
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Division by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        self.last_result = result
        return result
    
    def power(self, base: float, exponent: float) -> float:
        """Calculate power."""
        result = base ** exponent
        self.history.append(f"{base} ^ {exponent} = {result}")
        self.last_result = result
        return result
    
    def square_root(self, number: float) -> float:
        """Calculate square root."""
        if number < 0:
            raise ValueError("Square root of negative number")
        result = math.sqrt(number)
        self.history.append(f"sqrt({number}) = {result}")
        self.last_result = result
        return result
    
    def clear_history(self):
        """Clear the calculation history."""
        self.history = []
    
    def get_history(self) -> List[str]:
        """Get the calculation history."""
        return self.history.copy()

def generate_sample_data():
    """Generate sample data for testing purposes."""
    generator = DataGenerator(seed=42)
    
    # Generate sample people
    people = []
    for i in range(100):
        person = generator.random_person()
        people.append(person)
    
    # Generate sample numbers
    numbers = []
    for i in range(50):
        number = generator.random_integer(1, 1000)
        numbers.append(number)
    
    return people, numbers

def perform_math_operations():
    """Perform various mathematical operations."""
    math_util = MathUtilities()
    
    # Test basic operations
    result1 = math_util.add_numbers(10, 5)
    result2 = math_util.multiply_numbers(7, 8)
    result3 = math_util.power_function(2, 10)
    
    # Test factorial
    factorial_5 = math_util.factorial(5)
    
    # Test Fibonacci
    fib_sequence = math_util.fibonacci_sequence(10)
    
    # Test prime numbers
    primes = math_util.prime_numbers_up_to(50)
    
    return {
        'basic_ops': [result1, result2, result3],
        'factorial': factorial_5,
        'fibonacci': fib_sequence,
        'primes': primes
    }

def process_strings():
    """Process various strings using StringProcessor."""
    processor = StringProcessor()
    
    sample_text = "The quick brown fox jumps over the lazy dog"
    
    vowel_count = processor.count_vowels(sample_text)
    consonant_count = processor.count_consonants(sample_text)
    reversed_text = processor.reverse_string(sample_text)
    capitalized = processor.capitalize_words(sample_text)
    word_freq = processor.word_frequency(sample_text)
    is_palindrome = processor.is_palindrome("racecar")
    longest = processor.longest_word(sample_text)
    
    return {
        'vowels': vowel_count,
        'consonants': consonant_count,
        'reversed': reversed_text,
        'capitalized': capitalized,
        'frequency': word_freq,
        'palindrome_test': is_palindrome,
        'longest_word': longest
    }

def test_sorting_algorithms():
    """Test various sorting algorithms."""
    list_ops = ListOperations()
    
    # Create a random list to sort
    test_list = [64, 34, 25, 12, 22, 11, 90, 88, 76, 50, 42]
    
    bubble_sorted = list_ops.bubble_sort(test_list)
    selection_sorted = list_ops.selection_sort(test_list)
    insertion_sorted = list_ops.insertion_sort(test_list)
    
    # Test search algorithms
    sorted_list = sorted(test_list)
    binary_result = list_ops.binary_search(sorted_list, 25)
    linear_result = list_ops.linear_search(test_list, 25)
    
    # Test statistical functions
    maximum = list_ops.find_maximum(test_list)
    minimum = list_ops.find_minimum(test_list)
    average = list_ops.calculate_average(test_list)
    
    return {
        'original': test_list,
        'bubble_sort': bubble_sorted,
        'selection_sort': selection_sorted,
        'insertion_sort': insertion_sorted,
        'binary_search': binary_result,
        'linear_search': linear_result,
        'max': maximum,
        'min': minimum,
        'average': average
    }

def calculator_demo():
    """Demonstrate calculator functionality."""
    calc = Calculator()
    
    # Perform various calculations
    result1 = calc.add(15, 25)
    result2 = calc.multiply(result1, 2)
    result3 = calc.subtract(result2, 10)
    result4 = calc.divide(result3, 5)
    result5 = calc.power(result4, 2)
    result6 = calc.square_root(result5)
    
    history = calc.get_history()
    
    return {
        'final_result': result6,
        'calculation_history': history
    }

def time_operations():
    """Time various operations to demonstrate performance."""
    start_time = time.time()
    
    # Generate large dataset
    generator = DataGenerator()
    large_list = [generator.random_integer(1, 10000) for _ in range(5000)]
    
    generation_time = time.time() - start_time
    
    # Time sorting
    list_ops = ListOperations()
    
    start_sort = time.time()
    sorted_list = sorted(large_list)
    builtin_sort_time = time.time() - start_sort
    
    start_bubble = time.time()
    # Only sort first 100 elements for bubble sort (it's slow)
    bubble_sorted = list_ops.bubble_sort(large_list[:100])
    bubble_sort_time = time.time() - start_bubble
    
    return {
        'data_generation_time': generation_time,
        'builtin_sort_time': builtin_sort_time,
        'bubble_sort_time': bubble_sort_time,
        'list_size': len(large_list)
    }

def create_nested_data_structure():
    """Create a complex nested data structure."""
    data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'author': 'Python Script Generator'
        },
        'statistics': {
            'total_lines': 1200,
            'total_functions': 50,
            'total_classes': 8
        },
        'sample_data': {
            'numbers': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
            'strings': ['alpha', 'beta', 'gamma', 'delta', 'epsilon'],
            'booleans': [True, False, True, True, False]
        },
        'nested_structure': {
            'level_1': {
                'level_2': {
                    'level_3': {
                        'deep_value': 'Found at level 3!'
                    }
                }
            }
        }
    }
    
    return data

def validate_data_structure(data: Dict[str, Any]) -> bool:
    """Validate the structure of a data dictionary."""
    required_keys = ['metadata', 'statistics', 'sample_data']
    
    for key in required_keys:
        if key not in data:
            return False
    
    # Check metadata structure
    metadata = data.get('metadata', {})
    if 'created_at' not in metadata or 'version' not in metadata:
        return False
    
    # Check statistics structure
    statistics = data.get('statistics', {})
    required_stats = ['total_lines', 'total_functions', 'total_classes']
    for stat in required_stats:
        if stat not in statistics:
            return False
    
    return True

def process_json_data():
    """Process JSON data structures."""
    sample_json = {
        "users": [
            {"id": 1, "name": "Alice", "active": True},
            {"id": 2, "name": "Bob", "active": False},
            {"id": 3, "name": "Charlie", "active": True}
        ],
        "settings": {
            "theme": "dark",
            "notifications": True,
            "language": "en"
        }
    }
    
    # Convert to JSON string and back
    json_string = json.dumps(sample_json, indent=2)
    parsed_json = json.loads(json_string)
    
    # Extract active users
    active_users = [user for user in parsed_json['users'] if user['active']]
    
    return {
        'original': sample_json,
        'json_string': json_string,
        'active_users': active_users
    }

def fibonacci_iterative(n: int) -> int:
    """Calculate nth Fibonacci number iteratively."""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

def fibonacci_recursive(n: int) -> int:
    """Calculate nth Fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def compare_fibonacci_methods():
    """Compare iterative and recursive Fibonacci implementations."""
    test_values = [10, 15, 20, 25]
    results = {}
    
    for n in test_values:
        start_time = time.time()
        iterative_result = fibonacci_iterative(n)
        iterative_time = time.time() - start_time
        
        start_time = time.time()
        recursive_result = fibonacci_recursive(n)
        recursive_time = time.time() - start_time
        
        results[n] = {
            'iterative': {'result': iterative_result, 'time': iterative_time},
            'recursive': {'result': recursive_result, 'time': recursive_time}
        }
    
    return results

def matrix_operations():
    """Perform basic matrix operations."""
    # Create sample matrices
    matrix_a = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    matrix_b = [
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1]
    ]
    
    # Matrix addition
    result_add = []
    for i in range(len(matrix_a)):
        row = []
        for j in range(len(matrix_a[i])):
            row.append(matrix_a[i][j] + matrix_b[i][j])
        result_add.append(row)
    
    # Matrix transpose
    transpose_a = []
    for j in range(len(matrix_a[0])):
        row = []
        for i in range(len(matrix_a)):
            row.append(matrix_a[i][j])
        transpose_a.append(row)
    
    return {
        'matrix_a': matrix_a,
        'matrix_b': matrix_b,
        'addition': result_add,
        'transpose_a': transpose_a
    }

def password_strength_checker(password: str) -> Dict[str, Any]:
    """Check the strength of a password."""
    strength_score = 0
    feedback = []
    
    # Check length
    if len(password) >= 8:
        strength_score += 1
    else:
        feedback.append("Password should be at least 8 characters long")
    
    # Check for uppercase letters
    if any(c.isupper() for c in password):
        strength_score += 1
    else:
        feedback.append("Password should contain uppercase letters")
    
    # Check for lowercase letters
    if any(c.islower() for c in password):
        strength_score += 1
    else:
        feedback.append("Password should contain lowercase letters")
    
    # Check for digits
    if any(c.isdigit() for c in password):
        strength_score += 1
    else:
        feedback.append("Password should contain numbers")
    
    # Check for special characters
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    if any(c in special_chars for c in password):
        strength_score += 1
    else:
        feedback.append("Password should contain special characters")
    
    # Determine strength level
    if strength_score <= 2:
        strength_level = "Weak"
    elif strength_score <= 3:
        strength_level = "Medium"
    elif strength_score <= 4:
        strength_level = "Strong"
    else:
        strength_level = "Very Strong"
    
    return {
        'score': strength_score,
        'max_score': 5,
        'strength_level': strength_level,
        'feedback': feedback
    }

def text_statistics(text: str) -> Dict[str, Any]:
    """Calculate various statistics about a text."""
    words = text.split()
    sentences = text.split('.')
    paragraphs = text.split('\n\n')
    
    # Character counts
    total_chars = len(text)
    chars_no_spaces = len(text.replace(' ', ''))
    
    # Word statistics
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Sentence statistics
    sentence_count = len([s for s in sentences if s.strip()])
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    
    # Reading time estimation (average 200 words per minute)
    reading_time_minutes = word_count / 200
    
    return {
        'characters': total_chars,
        'characters_no_spaces': chars_no_spaces,
        'words': word_count,
        'sentences': sentence_count,
        'paragraphs': len(paragraphs),
        'avg_word_length': round(avg_word_length, 2),
        'avg_words_per_sentence': round(avg_words_per_sentence, 2),
        'estimated_reading_time': round(reading_time_minutes, 2)
    }

def currency_converter():
    """Simple currency converter with hardcoded exchange rates."""
    # Sample exchange rates (not real-time)
    exchange_rates = {
        'USD': {'EUR': 0.85, 'GBP': 0.73, 'JPY': 110.0, 'CAD': 1.25},
        'EUR': {'USD': 1.18, 'GBP': 0.86, 'JPY': 129.5, 'CAD': 1.47},
        'GBP': {'USD': 1.37, 'EUR': 1.16, 'JPY': 150.8, 'CAD': 1.71},
        'JPY': {'USD': 0.009, 'EUR': 0.008, 'GBP': 0.007, 'CAD': 0.011},
        'CAD': {'USD': 0.80, 'EUR': 0.68, 'GBP': 0.58, 'JPY': 88.0}
    }
    
    def convert(amount: float, from_currency: str, to_currency: str) -> float:
        """Convert amount from one currency to another."""
        if from_currency == to_currency:
            return amount
        
        if from_currency in exchange_rates and to_currency in exchange_rates[from_currency]:
            rate = exchange_rates[from_currency][to_currency]
            return amount * rate
        else:
            raise ValueError(f"Conversion not supported from {from_currency} to {to_currency}")
    
    # Perform sample conversions
    conversions = []
    conversions.append(('100 USD to EUR', convert(100, 'USD', 'EUR')))
    conversions.append(('50 EUR to GBP', convert(50, 'EUR', 'GBP')))
    conversions.append(('1000 JPY to USD', convert(1000, 'JPY', 'USD')))
    conversions.append(('200 CAD to EUR', convert(200, 'CAD', 'EUR')))
    
    return {
        'exchange_rates': exchange_rates,
        'sample_conversions': conversions
    }

def temperature_converter():
    """Convert temperatures between different scales."""
    
    def celsius_to_fahrenheit(celsius: float) -> float:
        """Convert Celsius to Fahrenheit."""
        return (celsius * 9/5) + 32
    
    def fahrenheit_to_celsius(fahrenheit: float) -> float:
        """Convert Fahrenheit to Celsius."""
        return (fahrenheit - 32) * 5/9
    
    def celsius_to_kelvin(celsius: float) -> float:
        """Convert Celsius to Kelvin."""
        return celsius + 273.15
    
    def kelvin_to_celsius(kelvin: float) -> float:
        """Convert Kelvin to Celsius."""
        return kelvin - 273.15
    
    def fahrenheit_to_kelvin(fahrenheit: float) -> float:
        """Convert Fahrenheit to Kelvin."""
        celsius = fahrenheit_to_celsius(fahrenheit)
        return celsius_to_kelvin(celsius)
    
    def kelvin_to_fahrenheit(kelvin: float) -> float:
        """Convert Kelvin to Fahrenheit."""
        celsius = kelvin_to_celsius(kelvin)
        return celsius_to_fahrenheit(celsius)
    
    # Sample temperature conversions
    conversions = {
        'water_freezing': {
            'celsius': 0,
            'fahrenheit': celsius_to_fahrenheit(0),
            'kelvin': celsius_to_kelvin(0)
        },
        'water_boiling': {
            'celsius': 100,
            'fahrenheit': celsius_to_fahrenheit(100),
            'kelvin': celsius_to_kelvin(100)
        },
        'room_temperature': {
            'celsius': 20,
            'fahrenheit': celsius_to_fahrenheit(20),
            'kelvin': celsius_to_kelvin(20)
        },
        'absolute_zero': {
            'celsius': kelvin_to_celsius(0),
            'fahrenheit': kelvin_to_fahrenheit(0),
            'kelvin': 0
        }
    }
    
    return conversions

def unit_converter():
    """Convert between various units of measurement."""
    
    # Length conversions (to meters)
    length_to_meters = {
        'mm': 0.001, 'cm': 0.01, 'm': 1, 'km': 1000,
        'inch': 0.0254, 'ft': 0.3048, 'yard': 0.9144, 'mile': 1609.34
    }
    
    # Weight conversions (to grams)
    weight_to_grams = {
        'mg': 0.001, 'g': 1, 'kg': 1000,
        'oz': 28.3495, 'lb': 453.592
    }
    
    # Volume conversions (to liters)
    volume_to_liters = {
        'ml': 0.001, 'l': 1,
        'fl_oz': 0.0295735, 'cup': 0.236588, 'pint': 0.473176,
        'quart': 0.946353, 'gallon': 3.78541
    }
    
    def convert_length(value: float, from_unit: str, to_unit: str) -> float:
        """Convert length units."""
        meters = value * length_to_meters[from_unit]
        return meters / length_to_meters[to_unit]
    
    def convert_weight(value: float, from_unit: str, to_unit: str) -> float:
        """Convert weight units."""
        grams = value * weight_to_grams[from_unit]
        return grams / weight_to_grams[to_unit]
    
    def convert_volume(value: float, from_unit: str, to_unit: str) -> float:
        """Convert volume units."""
        liters = value * volume_to_liters[from_unit]
        return liters / volume_to_liters[to_unit]
    
    # Sample conversions
    sample_conversions = {
        'length': [
            ('10 km to miles', convert_length(10, 'km', 'mile')),
            ('6 ft to meters', convert_length(6, 'ft', 'm')),
            ('100 cm to inches', convert_length(100, 'cm', 'inch'))
        ],
        'weight': [
            ('5 kg to pounds', convert_weight(5, 'kg', 'lb')),
            ('16 oz to grams', convert_weight(16, 'oz', 'g')),
            ('2.5 lb to kg', convert_weight(2.5, 'lb', 'kg'))
        ],
        'volume': [
            ('2 gallons to liters', convert_volume(2, 'gallon', 'l')),
            ('500 ml to fl_oz', convert_volume(500, 'ml', 'fl_oz')),
            ('3 cups to ml', convert_volume(3, 'cup', 'ml'))
        ]
    }
    
    return sample_conversions

def demonstrate_all_functionality():
    """Demonstrate all the functionality in this file."""
    results = {}
    
    print("Starting comprehensive demonstration...")
    
    # Generate sample data
    print("Generating sample data...")
    people, numbers = generate_sample_data()
    results['sample_data'] = {'people_count': len(people), 'numbers_count': len(numbers)}
    
    # Perform math operations
    print("Performing mathematical operations...")
    math_results = perform_math_operations()
    results['math_operations'] = math_results
    
    # Process strings
    print("Processing strings...")
    string_results = process_strings()
    results['string_processing'] = string_results
    
    # Test sorting algorithms
    print("Testing sorting algorithms...")
    sorting_results = test_sorting_algorithms()
    results['sorting_algorithms'] = sorting_results
    
    # Calculator demo
    print("Running calculator demo...")
    calc_results = calculator_demo()
    results['calculator_demo'] = calc_results
    
    # Time operations
    print("Timing operations...")
    timing_results = time_operations()
    results['timing_results'] = timing_results
    
    # Create nested data structure
    print("Creating nested data structure...")
    nested_data = create_nested_data_structure()
    is_valid = validate_data_structure(nested_data)
    results['nested_data'] = {'valid': is_valid, 'keys': list(nested_data.keys())}
    
    # Process JSON data
    print("Processing JSON data...")
    json_results = process_json_data()
    results['json_processing'] = {'active_users_count': len(json_results['active_users'])}
    
    # Compare Fibonacci methods
    print("Comparing Fibonacci implementations...")
    fib_comparison = compare_fibonacci_methods()
    results['fibonacci_comparison'] = fib_comparison
    
    # Matrix operations
    print("Performing matrix operations...")
    matrix_results = matrix_operations()
    results['matrix_operations'] = {'matrices_processed': 2}
    
    # Password strength check
    print("Checking password strength...")
    test_passwords = ["password", "Password123", "P@ssw0rd123!"]
    password_results = []
    for pwd in test_passwords:
        strength = password_strength_checker(pwd)
        password_results.append({
            'password': pwd,
            'strength': strength['strength_level'],
            'score': strength['score']
        })
    results['password_strength'] = password_results
    
    # Text statistics
    print("Calculating text statistics...")
    sample_text = """
    This is a sample text for analysis. It contains multiple sentences.
    We can calculate various statistics from this text.
    The analysis includes word count, character count, and reading time estimation.
    """
    text_stats = text_statistics(sample_text)
    results['text_statistics'] = text_stats
    
    # Currency conversion
    print("Testing currency conversion...")
    currency_results = currency_converter()
    results['currency_conversion'] = {'conversions_count': len(currency_results['sample_conversions'])}
    
    # Temperature conversion
    print("Testing temperature conversion...")
    temp_results = temperature_converter()
    results['temperature_conversion'] = {'scales_converted': len(temp_results)}
    
    # Unit conversion
    print("Testing unit conversion...")
    unit_results = unit_converter()
    results['unit_conversion'] = {
        'length_conversions': len(unit_results['length']),
        'weight_conversions': len(unit_results['weight']),
        'volume_conversions': len(unit_results['volume'])
    }
    
    print("Demonstration completed successfully!")
    return results

def create_sample_report():
    """Create a sample report with various data."""
    report = {
        'title': 'Sample Data Analysis Report',
        'generated_at': datetime.now().isoformat(),
        'sections': [
            {
                'name': 'Executive Summary',
                'content': 'This report demonstrates various Python capabilities including data processing, mathematical operations, and string manipulation.'
            },
            {
                'name': 'Data Overview',
                'content': 'The analysis includes sample datasets with randomly generated people and numerical data.'
            },
            {
                'name': 'Key Findings',
                'content': 'All implemented algorithms and utilities function correctly and efficiently process the test data.'
            },
            {
                'name': 'Recommendations',
                'content': 'The codebase provides a solid foundation for further development and can be extended with additional functionality.'
            }
        ],
        'appendices': [
            'Mathematical Operations Results',
            'String Processing Outputs',
            'Algorithm Performance Metrics',
            'Data Validation Results'
        ]
    }
    
    return report

# Main execution block
if __name__ == "__main__":
    """Main execution when script is run directly."""
    
    print("=" * 60)
    print("PYTHON DEMONSTRATION SCRIPT")
    print("=" * 60)
    print()
    
    # Initialize timing
    script_start_time = time.time()
    
    try:
        # Run comprehensive demonstration
        demo_results = demonstrate_all_functionality()
        
        # Create and display report
        report = create_sample_report()
        
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        
        print(f"Script execution time: {time.time() - script_start_time:.4f} seconds")
        print(f"Total demonstration modules: {len(demo_results)}")
        
        # Display key metrics
        if 'timing_results' in demo_results:
            timing = demo_results['timing_results']
            print(f"Data generation time: {timing['data_generation_time']:.4f} seconds")
            print(f"Built-in sort time: {timing['builtin_sort_time']:.4f} seconds")
            print(f"List size processed: {timing['list_size']} elements")
        
        if 'sample_data' in demo_results:
            data_info = demo_results['sample_data']
            print(f"Sample people generated: {data_info['people_count']}")
            print(f"Sample numbers generated: {data_info['numbers_count']}")
        
        print("\nAll operations completed successfully!")
        print("This script demonstrates comprehensive Python functionality.")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        sys.exit(1)
    
    finally:
        print("\nScript execution finished.")
        print("=" * 60)

# End of file - Total lines should be approximately 1200
