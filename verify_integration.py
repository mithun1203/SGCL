"""
Quick verification test for SG-CL training integration.

Tests that all components load and integrate correctly WITHOUT downloading models.
"""

import sys
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from sgcl_training import SGCLTrainer, NaiveFinetuningTrainer, TrainingConfig
    print("✓ sgcl_training imports successful")
except Exception as e:
    print(f"✗ sgcl_training import failed: {e}")
    sys.exit(1)

try:
    from sgcl_data_loader import SeCALoader, create_toy_tasks, create_minimal_tasks
    print("✓ sgcl_data_loader imports successful")
except Exception as e:
    print(f"✗ sgcl_data_loader import failed: {e}")
    sys.exit(1)

try:
    from sid import SemanticInconsistencyDetector
    print("✓ SID import successful")
except Exception as e:
    print(f"✗ SID import failed: {e}")
    sys.exit(1)

try:
    from guardrail import GuardrailController
    print("✓ Guardrail import successful")
except Exception as e:
    print(f"✗ Guardrail import failed: {e}")
    sys.exit(1)

# Test data loading
print("\nTesting data loaders...")
try:
    tasks, names = create_minimal_tasks()
    assert len(tasks) == 2, f"Expected 2 tasks, got {len(tasks)}"
    assert len(names) == 2, f"Expected 2 names, got {len(names)}"
    print(f"✓ Minimal tasks: {len(tasks)} tasks, {sum(len(t) for t in tasks)} samples")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

try:
    tasks, names = create_toy_tasks()
    assert len(tasks) == 3, f"Expected 3 tasks, got {len(tasks)}"
    print(f"✓ Toy tasks: {len(tasks)} tasks, {sum(len(t) for t in tasks)} samples")
except Exception as e:
    print(f"✗ Toy tasks failed: {e}")
    sys.exit(1)

# Test configuration
print("\nTesting configuration...")
try:
    config = TrainingConfig(
        model_name="microsoft/phi-3-mini-4k-instruct",
        lora_r=8,
        learning_rate=2e-4,
        max_guardrails=4
    )
    print(f"✓ TrainingConfig created")
    print(f"  - Model: {config.model_name}")
    print(f"  - LoRA r: {config.lora_r}")
    print(f"  - Max guardrails: {config.max_guardrails}")
except Exception as e:
    print(f"✗ Configuration failed: {e}")
    sys.exit(1)

# Test guardrail controller initialization (doesn't need model)
print("\nTesting guardrail controller...")
try:
    controller = GuardrailController(max_guardrails=4)
    print("✓ GuardrailController initialized")
    
    # Test with minimal batch
    kb = ["Birds can fly.", "Penguins are birds."]
    batch = ["Penguins cannot fly."]
    result = controller.process_batch(batch, kb)
    
    print(f"  - Conflict detected: {result.has_conflict}")
    print(f"  - Guardrails added: {len(result.guardrail_samples)}")
    if result.guardrail_samples:
        print(f"  - Example guardrail: {result.guardrail_samples[0]}")
except Exception as e:
    print(f"✗ Guardrail controller failed: {e}")
    sys.exit(1)

# Test SID
print("\nTesting SID...")
try:
    sid = SemanticInconsistencyDetector()
    conflict = sid.detect_conflict("Penguins can fly.")
    print(f"✓ SID initialized and working")
    print(f"  - Test conflict result: {conflict.has_conflict}")
except Exception as e:
    print(f"✗ SID failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL VERIFICATION TESTS PASSED!")
print("="*70)
print("\nYour SG-CL system is ready for training!")
print("\nNext steps:")
print("1. Run minimal experiment: python run_sgcl_experiment.py --dataset minimal")
print("2. This will download Phi-3 model (~7.5GB) on first run")
print("3. Training will take ~5-10 minutes on CPU (faster on GPU)")
