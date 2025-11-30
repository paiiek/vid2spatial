# Vid2Spatial Test Suite

Comprehensive test suite for the vid2spatial module.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── test_vision.py           # Vision module tests (camera geometry, tracking)
├── test_foa_render.py       # Spatial audio encoding tests
└── test_integration.py      # End-to-end pipeline tests
```

## Running Tests

### Run all tests
```bash
cd mmhoa/vid2spatial
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_vision.py
pytest tests/test_foa_render.py
pytest tests/test_integration.py
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run specific test class or function
```bash
pytest tests/test_vision.py::TestCameraIntrinsics
pytest tests/test_foa_render.py::TestFOAGainCalculation::test_front_direction
```

### Skip slow tests
```bash
pytest tests/ -m "not slow"
```

### Run only integration tests
```bash
pytest tests/ -m integration
```

### Generate coverage report
```bash
pytest tests/ --cov=mmhoa.vid2spatial --cov-report=html
```

## Test Categories

### Unit Tests

**test_vision.py**
- `TestCameraIntrinsics`: Camera parameter calculations
- `TestPixelToRay`: Pixel to 3D ray conversion
- `TestRayToAngles`: Ray to spherical angle conversion
- `TestPixelToAnglesPipeline`: Combined pixel → ray → angles
- `TestEdgeCases`: Edge cases and error handling
- `TestCameraGeometryConsistency`: Geometric consistency checks

**test_foa_render.py**
- `TestFOAGainCalculation`: AmbiX FOA gain encoding
- `TestAngleInterpolation`: Trajectory interpolation
- `TestAngleSmoothing`: Smoothing and delta limiting
- `TestMonoToFOAEncoding`: Mono to FOA conversion
- `TestDistanceRendering`: Distance-based gain and filtering
- `TestEdgeCases`: Boundary conditions

### Integration Tests

**test_integration.py**
- `TestPixelToSpatialPipeline`: Complete pixel → spatial audio flow
- `TestSpatialConsistency`: Spatial encoding consistency
- `TestDistanceEffects`: Distance attenuation and filtering
- `TestAngleInterpolation`: Trajectory interpolation quality
- `TestEndToEndScenarios`: Real-world usage scenarios
- `TestPerformance`: Performance characteristics (marked as slow)

## Fixtures

Available in `conftest.py`:
- `sample_audio_mono`: 1-second mono audio at 48kHz (440 Hz sine)
- `sample_trajectory`: Synthetic 3D trajectory (30 frames, moving left-to-right)
- `sample_camera_intrinsics`: Standard 1920×1080 camera with 60° FOV
- `test_data_dir`: Temporary directory for test data
- `synthetic_video`: Generated video with moving rectangle
- `synthetic_trajectory`: 3D trajectory matching synthetic video

## Requirements

```bash
pip install pytest pytest-cov numpy opencv-python soundfile
```

## Test Coverage Goals

- **Vision module**: >90% coverage
  - Core geometry functions: 100%
  - Tracking functions: >85%
  - Depth estimation: >80%

- **FOA render module**: >95% coverage
  - Encoding functions: 100%
  - Interpolation: 100%
  - Distance effects: >90%

- **Integration tests**: Cover all major workflows
  - Static source encoding
  - Moving source panning
  - Distance effects
  - Complete pipeline

## Adding New Tests

### Test Naming Convention
- Test files: `test_<module>.py`
- Test classes: `Test<Feature>`
- Test functions: `test_<specific_behavior>`

### Example Test

```python
def test_my_new_feature(sample_camera_intrinsics):
    """Test description goes here."""
    # Arrange
    K = sample_camera_intrinsics
    input_data = ...

    # Act
    result = my_function(input_data, K)

    # Assert
    assert result.shape == expected_shape
    assert np.all(np.isfinite(result))
```

### Using Markers

```python
@pytest.mark.slow
def test_expensive_operation():
    """This test takes a long time."""
    ...

@pytest.mark.integration
def test_full_pipeline():
    """Integration test for complete workflow."""
    ...

@pytest.mark.requires_gpu
def test_gpu_acceleration():
    """This test requires GPU."""
    ...
```

## Continuous Integration

Tests should pass on:
- Python 3.8+
- NumPy 1.20+
- OpenCV 4.5+

## Troubleshooting

### Import errors
Make sure you're in the correct directory:
```bash
cd mmhoa/vid2spatial
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../.."
pytest tests/
```

### Missing dependencies
```bash
pip install -e ../../  # Install mmhoa package in editable mode
```

### Video creation fails
On systems without video codec support, synthetic video tests may fail. Skip with:
```bash
pytest tests/ --ignore=tests/test_integration.py::TestEndToEndScenarios::test_static_front_source
```
