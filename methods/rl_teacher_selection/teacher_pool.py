"""Teacher pool management with unified caching.

This module provides:
- Teacher: Wrapper for individual teacher models
- TeacherPool: Manages multiple teachers with unified cache

Cache design:
- preloaded_cache: Data loaded from files (avoids inference, but cost = teacher.cost)
- runtime_cache: Data generated in current run (second access has cost = 0)
"""

import torch
from typing import List, Tuple, Dict, Any, Optional


class Teacher:
    """Wrapper for a teacher model."""

    def __init__(
        self,
        name: str,
        model: Any,
        cost: float,
        generate_fn: Optional[callable] = None,
    ):
        """Initialize teacher.

        Args:
            name: Teacher name (e.g., "gpt-4")
            model: Teacher model object
            cost: Cost of calling this teacher
            generate_fn: Optional custom generation function
                         If None, will call model.generate(instruction)
        """
        self.name = name
        self.model = model
        self.cost = cost
        self.generate_fn = generate_fn

    def generate(self, instruction: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate response for instruction.

        Args:
            instruction: Input instruction

        Returns:
            (input_ids, labels) tuple
        """
        if self.generate_fn is not None:
            return self.generate_fn(self.model, instruction)
        else:
            return self.model.generate(instruction)


class TeacherPool:
    """Manages multiple teachers with unified caching."""

    def __init__(self, teachers: List[Teacher]):
        """Initialize teacher pool.

        Args:
            teachers: List of Teacher objects
        """
        self.teachers = teachers
        self.teacher_names = [t.name for t in teachers]

        # Preloaded cache: loaded from files, avoids inference but cost = teacher.cost
        self.preloaded_cache: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {
            name: {} for name in self.teacher_names
        }

        # Runtime cache: generated in current run, second access has cost = 0
        self.runtime_cache: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {
            name: {} for name in self.teacher_names
        }

        # Cache statistics
        self.cache_hits = {name: 0 for name in self.teacher_names}
        self.cache_misses = {name: 0 for name in self.teacher_names}

    def generate(
        self, teacher_idx: int, instruction: str
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float]:
        """Generate response using specified teacher.

        Args:
            teacher_idx: Index of teacher to use
            instruction: Input instruction

        Returns:
            ((input_ids, labels), cost) tuple

        Cost logic:
            - If in runtime_cache: cost = 0 (true reuse)
            - If in preloaded_cache: cost = teacher.cost (avoids inference, but counts as "calling teacher")
            - Otherwise: cost = teacher.cost (actual generation)
        """
        teacher = self.teachers[teacher_idx]
        teacher_name = teacher.name

        # Check runtime cache first (cost = 0)
        if instruction in self.runtime_cache[teacher_name]:
            self.cache_hits[teacher_name] += 1
            response = self.runtime_cache[teacher_name][instruction]
            return response, 0.0

        # Check preloaded cache (cost = teacher.cost, but no inference)
        if instruction in self.preloaded_cache[teacher_name]:
            response = self.preloaded_cache[teacher_name][instruction]
            # Add to runtime cache for next access
            self.runtime_cache[teacher_name][instruction] = response
            return response, teacher.cost

        # Cache miss - generate new response
        self.cache_misses[teacher_name] += 1
        response = teacher.generate(instruction)

        # Store in runtime cache
        self.runtime_cache[teacher_name][instruction] = response

        return response, teacher.cost

    def load_preloaded_cache(
        self, teacher_idx: int, cache_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ):
        """Load preloaded cache for a teacher.

        Args:
            teacher_idx: Index of teacher
            cache_data: Dictionary mapping instruction to (input_ids, labels)
        """
        teacher_name = self.teacher_names[teacher_idx]
        self.preloaded_cache[teacher_name].update(cache_data)

    def prefill_cache(
        self, teacher_idx: int, instructions: List[str]
    ):
        """Prefill cache for a teacher by generating responses.

        Args:
            teacher_idx: Index of teacher
            instructions: List of instructions to prefill
        """
        teacher = self.teachers[teacher_idx]
        teacher_name = teacher.name

        for instruction in instructions:
            if instruction not in self.runtime_cache[teacher_name] and \
               instruction not in self.preloaded_cache[teacher_name]:
                response = teacher.generate(instruction)
                self.runtime_cache[teacher_name][instruction] = response

    def get_cache_stats(self, teacher_idx: Optional[int] = None) -> Dict:
        """Get cache statistics.

        Args:
            teacher_idx: If specified, return stats for this teacher only
                        If None, return stats for all teachers

        Returns:
            Dictionary with cache statistics
        """
        if teacher_idx is not None:
            teacher_name = self.teacher_names[teacher_idx]
            total = self.cache_hits[teacher_name] + self.cache_misses[teacher_name]
            hit_rate = self.cache_hits[teacher_name] / total if total > 0 else 0.0

            return {
                "teacher": teacher_name,
                "preloaded_cache_size": len(self.preloaded_cache[teacher_name]),
                "runtime_cache_size": len(self.runtime_cache[teacher_name]),
                "cache_hits": self.cache_hits[teacher_name],
                "cache_misses": self.cache_misses[teacher_name],
                "hit_rate": hit_rate,
            }
        else:
            # Return stats for all teachers
            stats = {}
            for idx, name in enumerate(self.teacher_names):
                stats[name] = self.get_cache_stats(idx)
            return stats

    def clear_cache(self, teacher_idx: Optional[int] = None):
        """Clear cache.

        Args:
            teacher_idx: If specified, clear cache for this teacher only
                        If None, clear all caches
        """
        if teacher_idx is not None:
            teacher_name = self.teacher_names[teacher_idx]
            self.preloaded_cache[teacher_name].clear()
            self.runtime_cache[teacher_name].clear()
            self.cache_hits[teacher_name] = 0
            self.cache_misses[teacher_name] = 0
        else:
            for name in self.teacher_names:
                self.preloaded_cache[name].clear()
                self.runtime_cache[name].clear()
                self.cache_hits[name] = 0
                self.cache_misses[name] = 0

    def __len__(self) -> int:
        """Return number of teachers."""
        return len(self.teachers)

    def __getitem__(self, idx: int) -> Teacher:
        """Get teacher by index."""
        return self.teachers[idx]
