(type-confusion)=
Type Confusion
==============

When using the ndindex API, it is important to avoid type confusion. Many
types that are used as indices for arrays also have semantic meaning outside
of indexing. For example, tuples and arrays mean one thing when they are
indices, but they are also used in contexts that have nothing to do with indexing.

ndindex classes have names that are based on the native class names for the
index type they represent. One must be careful, however, to not confuse these
classes with the classes they represent. Most methods that work on the native
classes are not available on the ndindex classes.

Some general types to help avoid type confusion:

- **Always use the [`ndindex()`](ndindex.ndindex) function to create ndindex
  types.** When calling ndindex methods or creating `Tuple
  <ndindex.tuple.Tuple>` objects, it is not necessary to convert arguments to
  ndindex types first. Slice literals (using `:`) are not valid syntax outside
  of a getitem (square brackets), but you can use the `slice` built-in object
  to create slices. `slice(a, b, c)` is the same as `a:b:c`.

  **Right:**

  ```py
  idx.as_subindex((1,))
  ```

  **Wrong:**

  ```
  idx.as_subindex(Tuple(Integer(1))) # More verbose than necessary
  ```


- **Use `.raw` to convert an ndindex object to an indexable type.** With the
  exception of `Integer`, it is impossible for custom types to define
  themselves as indices to NumPy arrays, so it is necessary to use
  `a[idx.raw]` rather than `a[idx]` when `idx` is an ndindex type. Since
  native index types do not have a `.raw` method, it is recommended to always
  keep any index object that you are using as an ndindex type, and use `.raw`
  only when you need to use it as an index. If you get the error "``IndexError:
  only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and
  integer or boolean arrays are valid indices``", it indicates you forgot to
  use `.raw`.

  **Right:**

  ```py
  a[idx.raw]
  ```

  **Wrong:**

  ```py
  a[idx] # Gives an error
  ```

- **Only use ndindex classes for objects that represent indices.** Do not use
  classes like `Integer`, `Tuple`, `IntegerArray`, or `BooleanArray` unless
  the object in question is going to be an index to an array. For example,
  array shapes are always tuples of integers, but they are not indices, so
  `Tuple` should not be used to represent an array shape, but rather just a
  normal `tuple`. If an object will be used both as an index and an
  integer/tuple/array in its own right, either make use of `.raw` when using
  it in non-index contexts, or store the objects separately (note that
  `IntegerArray` and `BooleanArray` always make a copy of the input argument,
  so there is no issue with managing it separately).

  **Right:**

  ```py
  np.empty((1, 2, 3))
  # idx is an ndindex object
  idx.newshape((1, 2, 3))
  ```

  **Wrong:**

  ```py
  np.empty(Tuple(1, 2, 3)) # gives an error
  # idx is an ndindex object
  idx.newshape(Tuple(1, 2, 3)) # gives an error
  ```

- **Try to use ndindex methods to manipulate indices.** The whole reason
  ndindex exists is that writing formulas for manipulating indices is hard,
  and it's easy to get the corner cases wrong. If you find yourself
  manipulating index args directly in complex ways, it's a sign you should
  probably be using a higher level abstraction. If what you are trying to do
  doesn't exist yet, [open an
  issue](https://github.com/Quansight/ndindex/issues) so we can implement it.

Additionally, some advice for specific types:

## Integer

- **{any}`Integer` should not be thought of as an int type.** It represents integers
  **as indices**. It is not usable in contexts where integers are usable. For
  example, arithmetic will not work on it. If you need to manipulate the
  integer index as an integer, use `idx.raw`.

  **Right:**

  ```py
  # idx is an Integer
  idx.raw + 1
  ```

  **Wrong:**

  ```py
  # idx is a Tuple
  idx.raw + 1 # Produces an error
  ```

- `Integer` is the only index type that can be used directly as an array
  index. This is because the `__index__` API allows custom integer objects to
  define themselves as indices. However, this API does not extend to other
  index types like slices or tuples. **It is recommended to always use
  `idx.raw` even if `idx` is an `Integer`**, so that it will also work even if
  it is another index type. You should not rely on any ndindex function
  returning a specific index type.

## Tuple

- **{any}`Tuple` should not be thought of as a tuple.** In particular, things like
  `idx[0]` and `len(idx)` will not work if `idx` is a `Tuple`. If you need to
  access the specific term in a `Tuple`, use `Tuple.args`.

  **Right:**

  ```py
  # idx is a Tuple
  idx.raw[0]
  ```

  **Wrong:**

  ```py
  # idx is a Tuple
  idx[0] # Produces an error
  ```

- `Tuple` is defined as `Tuple(*args)`.

   **Right:**

   ```py
   Tuple(0, 1, 2)
   ```

   **Wrong:**

   ```py
   Tuple((0, 1, 2)) # Gives an error
   ```

## ellipsis

- You should almost never use the ndindex {any}`ellipsis` class directly.
  Instead, **use `...` or `ndindex(...)`**. As noted above, all ndindex
  methods and `Tuple` will automatically convert `...` into the ndindex type.

  **Right:**

  ```py
  idx = ndindex(...)
  idx.reduce()
  ```

  **Wrong:**

  ```py
  idx = ...
  idx.reduce() # Gives an error
  ```

- If you do use `ellipsis` beware that it is the *class*, not the *instance*,
  unlike the built-in `Ellipsis` object. This is done for consistency in the
  internal ndindex class hierarchy.

  **Right:**

  ```py
  idx = ndindex((0, ..., 1))
  ```

  **Wrong:**

  ```py
  idx = ndindex((0, ellipsis, 1)) # Gives an error
  ```

  These do not give errors, but it is easy to confuse them with the above. It
  is best to just use `...`, which is more concise and easier to read.

  ```py
  idx = ndindex((0, ellipsis(), 1))
  idx.reduce()
  ```

  ```py
  idx = ndindex((0, Ellipsis, 1))
  idx.reduce()
  ```

- `ellipsis` is **not** singletonized, unlike the built-in `...`. It would
  also be impossible to make `ellipsis() is ...` return True. If you are using
  ndindex, **you should use `==` to compare against `...`**, and avoid using `is`.

  **Right:**

  ```py
  if idx == ...:
  ```

  **Wrong:**

  ```py
  if idx is Ellipsis: # Will be False if idx is the ndindex ellipsis type
  ```

  ```py
  if idx is ellipsis(): # Will be False (ellipsis() creates a new instance)
  ```

## Newaxis

The advice for `Newaxis` is almost identical to the advice for `ellipsis`.
Note that `np.newaxis` is just an alias for `None`.

- You should almost never use the ndindex {any}`Newaxis` class directly.
  Instead, **use `np.newaxis`, `None`, `ndindex(np.newaxis)`, or
  `ndindex(None)`**. As noted above, all ndindex methods and `Tuple` will
  automatically convert `None` into the ndindex type.

  **Right:**

  ```py
  idx = ndindex(np.newaxis)
  idx.reduce()
  ```

  **Wrong:**

  ```py
  idx = np.newaxis
  idx.reduce() # Gives an error
  ```

- If you do use `Newaxis` beware that it is the *class*, not the *instance*,
  unlike the NumPy `np.newaxis` object (i.e., `None`). This is done for
  consistency in the internal ndindex class hierarchy.

  **Right:**

  ```py
  idx = ndindex((0, np.newaxis, 1))
  ```

  **Wrong:**

  ```py
  idx = ndindex((0, Newaxis, 1)) # Gives an error
  ```

  This does not give an error, but it is easy to confuse it with the above. It
  is best to just use `np.newaxis` or `None`, which is more concise and easier
  to read.

  ```py
  idx = ndindex((0, Newaxis(), 1))
  idx.reduce()
  ```

- `Newaxis` is **not** singletonized, unlike the built-in `None`. It would
  also be impossible to make `Newaxis() is np.newaxis` or `Newaxis() is None`
  return True. If you are using ndindex, **you should use `==` to compare
  against `np.newaxis` or `None`**, and avoid using `is`.

  **Right:**

  ```py
  if idx == np.newaxis:
  ```

  **Wrong:**

  ```py
  if idx is np.newaxis: # Will be False if idx is the ndindex Newaxis type
  ```

  ```py
  if idx is Newaxis(): # Will be False (Newaxis() creates a new instance)
  ```

## IntegerArray and BooleanArray

- **{any}`IntegerArray` and `BooleanArray` should not be thought of as
  arrays.** They do not have the methods that `numpy.ndarray` would have. They
  also have fixed dtypes (`intp` and `bool_`) and are restricted by what is
  allowed as indices by NumPy.

  **Right:**

  ```py
  idx = IntegerArray(array([0, 1]))
  idx.array[0]
  ```

  **Wrong:**

  ```py
  idx = IntegerArray(array([0, 1]))
  idx.[0] # Gives an error
  ```

- **Like all other ndindex types, `IntegerArray` and `BooleanArray` are
  immutable.**. The `.array` object on them is set as read-only to enforce
  this. To modify an array index, create a new object. All ndindex methods
  that manipulate indices, like [reduce](NDIndex.reduce), return new objects.
  If you create an `IntegerArray` or `BooleanArray` object out of an existing
  array, the array is copied so that modifications to the original array do
  not affect the ndindex objects.

  **Right:**

  ```py
  idx = IntegerArray(array([0, 1]))
  arr = idx.array.copy()
  arr[0] = 1
  idx2 = IntegerArray(arr)
  ```

  **Wrong:**

  ```py
  idx = IntegerArray(array([0, 1]))
  idx.array[0] = 1 # Gives an error
  ```
