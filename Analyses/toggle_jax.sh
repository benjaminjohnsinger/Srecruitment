# find Analyses -name "*.py" -exec sed -i '' 's/import numpy as np/import jax.numpy as np/g' {} +
# find Analyses -name "*.py" -exec sed -i '' 's/import numpy as np/import jax.numpy as jnp/g' {} +
# find Analyses -name "*.py" -exec sed -i '' 's/np/jnp/g' {} +
find Analyses -name "*.py" -exec sed -i '' 's/from numba import jit//g' {} +