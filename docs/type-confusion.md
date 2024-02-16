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

- **Always use the {func}`~.ndindex` function to create ndindex types.** When
  calling ndindex methods or creating {class}`~.Tuple` objects, it is not
  necessary to convert arguments to ndindex types first. Slice literals (using
  `:`) are not valid syntax outside of a getitem (square brackets), but you
  can use the `slice` built-in object to create slices. `slice(a, b, c)` is
  the same as `a:b:c`.

  **Wrong:**

  ```
  idx.as_subindex(Tuple(Integer(1))) # More verbose than necessary
  ```

  **Right:**

  ```py
  idx.as_subindex((1,))
  ```

- **Keep all index objects as ndindex types until performing actual
  indexing.** If all you are doing with an index is indexing it, and not
  manipulating it or storing it somewhere else, there is no need to use
  ndindex. But if you are storing ndindex types separately before indexing, or
  plan to manipulate them in any way, it is best to convert them to ndindex
  types with [`ndindex()`](ndindex.ndindex) as early as possible, and store
  them in that form. Only convert back to a raw index (with `.raw`, see the
  next bullet point) once doing an actual index operation. Avoid mixing
  ndindex and "raw" or non-ndindex types. There are many reasons to avoid
  this:

  - Raw types (such as `int`, `slice`, `tuple`, `array`, and so on), do not
    have any of the same methods as ndindex, so your code may fail.

  - Some raw types, such as slices, arrays, and tuples containing slices or
    arrays, are not hashable, so if you try to use them as a dictionary key,
    they will fail. ndindex types are always hashable.

    **Wrong**

    ```py
    # Fails with a TypeError: unhashable type: 'slice'
    indices = {
        slice(0, 10): 0,
        slice(10, 20): 1
    }
    ```

    **Right**

    ```py
    indices = {
        ndindex[0:10]: 0,
        ndindex[10:20]: 1
    }
    ```

  - ndindex does basic type checking on indices that would otherwise not
    happen until they are actually used as an index. For example, `slice(1.0,
    2.0)` does not fail until you try to index an array with it, but
    `Slice(1.0, 2.0)` fails immediately.

    **Wrong**

    ```py
    # Typo would not be caught until idx is actually used to index an array
    idx = slice(1, 2.)
    ```

    **Right**

    ```py
    # Typo would be caught right away
    idx = ndindex[1:2.]
    # OR
    idx = Slice(1, 2.)
    ```

  - NumPy arrays and tuples containing NumPy arrays are not easy to compare,
    since using `==` on an `array` does not produce a boolean. `==` on an
    ndindex type will always produce a boolean, which compares if the two
    indices are exactly equal.

    **Wrong**

    ```py
    # Fails with ValueError: The truth value of an array with more than one element is ambiguous.
    if idx == (slice(0, 2), np.array([0, 0]):
        ...
    ```

    **Right**

    ```py
    # Note only one side of the == needs to be an ndindex type
    if ndindex(idx) == (slice(0, 2), np.array([0, 0]):
        ...
    ```

    Additionally, all ndindex types are immutable, including types
    representing NumPy arrays, so it is impossible to accidentally mutate an
    ndindex array index object.

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

- **Use ndindex classes only for objects that represent indices.** Do not use
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
  and it's easy to get the corner cases wrong. ndindex is [rigorously
  tested](testing) so you can be highly confident of its correctness. If you
  find yourself manipulating index args directly in complex ways, it's a sign
  you should probably be using a higher level abstraction. If what you are
  trying to do doesn't exist yet, [open an
  issue](https://github.com/Quansight-labs/ndindex/issues) so we can implement
  it.

Additionally, some advice for specific types:

## Integer

- **{class}`~.Integer` should not be thought of as an int type.** It
  represents integers **as indices**. It is not usable in other contexts where
  ints are usable. For example, arithmetic will not work on it. If you need to
  manipulate the `Integer` index as an `int`, use `idx.raw`.

  **Right:**

  ```py
  idx = ndindex(0)
  idx.raw + 1
  ```

  **Wrong:**

  ```py
  idx = ndindex(0)
  idx + 1 # Produces an error
  ```

- `Integer` is the only index type that can be used directly as an array
  index. This is because the `__index__` API allows custom integer objects to
  define themselves as indices. However, this API does not extend to other
  index types like slices or tuples. **It is recommended to always use
  `idx.raw` even if `idx` is an `Integer`**, so that it will also work even if
  it is another index type. You should not rely on any ndindex function
  returning a specific index type (unless it states that it does so in its
  docstring).

(type-confusion-tuples)=
## Tuple

- **{class}`~.Tuple` should not be thought of as a tuple.** In particular,
  things like `idx[0]` and `len(idx)` will not work if `idx` is a `Tuple`. If
  you need to access the specific term in a `Tuple`, use `Tuple.args` if you
  want the ndindex type, or `Tuple.raw` if you want the raw type.

  **Right:**

  ```py
  idx = ndindex[0, 0:1]
  idx.raw[0] # Gives int(0)
  idx.args[0] # Gives Integer(0)
  ```

  **Wrong:**

  ```py
  idx = ndindex[0, 0:1]
  idx[0] # Produces an error
  ```

- `Tuple` is defined as `Tuple(*args)`. `Tuple(args)` gives an error.

   **Right:**

   ```py
   Tuple(0, slice(0, 1))
   ```

   **Better:**
   ```py
   ndindex[0, 0:1]
   ```

   **Wrong:**

   ```py
   Tuple((0, slice(0, 1))) # Gives an error
   ```

(type-confusion-ellipsis)=
## ellipsis

- You should almost never use the ndindex {class}`~.ellipsis` class directly.
  Instead, **use `...` or `ndindex[...]`**. As noted above, all ndindex
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

- We recommend preferring `...` over the built-in name `Ellipsis`, as it is
  more readable. The `...` syntax is allowed everywhere that `Ellipsis` would
  work.

  **Right:**

  ```py
  idx = ndindex[0, ..., 1]
  ```

  **Wrong:**

  ```py
  idx = ndindex[0, Ellipsis, 1] # Less readable
  ```


- If you do use `ellipsis` beware that it is the *class*, not the *instance*,
  unlike the built-in `Ellipsis` object. This is done for consistency in the
  internal ndindex class hierarchy.

  **Right:**

  ```py
  idx = ndindex[0, ..., 1]
  ```

  **Wrong:**

  ```py
  idx = ndindex[0, ellipsis, 1] # Gives an error
  ```

  The below do not give errors, but it is easy to confuse them with the above.
  It is best to just use `...`, which is more concise and easier to read.

  ```py
  idx = ndindex[0, ellipsis(), 1] # Easy to confuse, less readable
  idx.reduce()
  ```

  ```py
  idx = ndindex[0, Ellipsis, 1] # Easy to confuse, less readable
  idx.reduce()
  ```

- `ellipsis` is **not** singletonized, unlike the built-in `...`. Aside from
  singletonization not being necessary for ndindex types, it would be
  impossible to make `ellipsis() is ...` return True. If you are using
  ndindex, **you should use `==` to compare against `...`**, and avoid using
  `is`. Note that as long as you know `idx` is an ndindex type, this is safe
  to do, since even the array index types `IntegerArray` and `BooleanArray`
  allow `==` comparison (unlike NumPy arrays).

  **Right:**

  ```py
  if idx == ...:
  ```

  **Wrong:**

  ```py
  if idx is Ellipsis: # Will be False if idx is the ndindex ellipsis type
  ```

  ```py
  if idx is ellipsis(): # Will always be False (ellipsis() creates a new instance)
  ```

## Newaxis

The advice for `Newaxis` is almost identical to the advice for
[`ellipsis`](type-confusion-ellipsis). Note that `np.newaxis` is just an alias
for `None`.

- You should almost never use the ndindex {class}`~.Newaxis` class directly.
  Instead, **use `np.newaxis`, `None`, `ndindex(np.newaxis)`, or
  `ndindex(None)`**. As noted above, all ndindex methods and `Tuple` will
  automatically convert `None` into the ndindex type.

  **Right:**

  ```py
  idx = ndindex[np.newaxis]
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
  idx = ndindex[0, np.newaxis, 1]
  ```

  ```py
  idx = ndindex[0, None, 1]
  ```

  **Wrong:**

  ```py
  idx = ndindex[0, Newaxis, 1] # Gives an error
  ```

  The below does not give an error, but it is easy to confuse it with the
  above. It is best to just use `np.newaxis` or `None`, which is more concise
  and easier to read.

  ```py
  idx = ndindex[0, Newaxis(), 1] # Easy to confuse
  idx.reduce()
  ```

- `Newaxis` is **not** singletonized, unlike the built-in `None`. It would
  also be impossible to make `Newaxis() is np.newaxis` or `Newaxis() is None`
  return True. If you are using ndindex, **you should use `==` to compare
  against `np.newaxis` or `None`**, and avoid using `is`. Note that as long as
  you know `idx` is an ndindex type, this is safe to do, since even the array
  index types `IntegerArray` and `BooleanArray` allow `==` comparison (unlike
  NumPy arrays).

  **Right:**

  ```py
  if idx == np.newaxis:
  ```

  ```py
  if idx == None:
  ```

  **Wrong:**

  ```py
  if idx is np.newaxis: # Will be False if idx is the ndindex Newaxis type
  ```

  ```py
  if idx is None: # Will be False if idx is the ndindex Newaxis type
  ```

  ```py
  if idx is Newaxis(): # Will be False (Newaxis() creates a new instance)
  ```

## IntegerArray and BooleanArray

- **{class}`~.IntegerArray` and {class}`~.BooleanArray` should not be thought
  of as arrays.** They do not have the methods that `numpy.ndarray` would
  have. They also have fixed dtypes (`intp` and `bool_`) and are restricted by
  what is allowed as indices by NumPy. To access the arrays they represent,
  use `idx.array` or `idx.raw`.

  **Right:**

  ```py
  idx = IntegerArray(np.array([0, 1]))
  idx.array[0]
  ```

  **Wrong:**

  ```py
  idx = IntegerArray(np.array([0, 1]))
  idx[0] # Gives an error
  ```

- **Like all other ndindex types, `IntegerArray` and `BooleanArray` are
  immutable.**. The `.array` object on them is set as read-only to enforce
  this. To modify an array index, create a new object. All ndindex methods
  that manipulate indices, like [`reduce()`](NDIndex.reduce), return new
  objects. If you create an `IntegerArray` or `BooleanArray` object out of an
  existing array, the array is copied so that modifications to the original
  array do not affect the ndindex objects.

  **Right:**

  ```py
  idx = IntegerArray(np.array([0, 1]))
  arr = idx.array.copy()
  arr[0] = 1
  idx2 = IntegerArray(arr)
  ```

  **Wrong:**

  ```py
  idx = IntegerArray(np.array([0, 1]))
  idx.array[0] = 1 # Gives an error
  ```
