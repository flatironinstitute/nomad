# Contributors Guide

Thanks for your interest in contributing to this project!

There are many ways to contribute, and we welcome contributions of all kinds:
 
 * Using and publicizing the library
 * Feedback, questions, bug reports, and feature requests
 * Improvements to documentation
 * Fixes for errors
 * Adding new algorithms

## Ways to Contribute

### Using and publicizing

Using the project *is* contributing to the project.

If you find it useful or apply it to a new type of problem, we'd love to
hear about it--please reach out.

Once the library is fully live, please feel free to publicize it through
any channels you normally use--in conversations or word-of-mouth, at
more formal presentations, on social media, and so on.

If this library has contributed materially to your published work,
please cite as CITATION TBD.

### Feedback and Issues

The best way to communicate with the developers about suggestions,
questions, challenges, or errors is through opening a 
[Github issue on this repository](https://github.com/flatironinstitute/nomad/issues).
This provides a public and long-lived record of the
issue that will help guide your issue to resolution, and serve as a
reference for future users.


### Suggesting Changes

Contributions that involve changing code or documentation are
also welcome. The standard mechanism for suggesting and reviewing
changes within a Github-based project is the
[pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

There are many strategies for organizing pull requests. For this
project, an ideal process would be as follows:

* **Open a Github issue**, if one doesn't exist already. The
issue is a place to define the problem clearly, establish the
scope of any proposed changes, and discuss the design of a good
solution. Even better, it provides a public record of these
discussions which can help guide new users and future work.
* **Fork the repository** if you do not have rights to create your
own branch on this repository.
* **Create a branch for your changes**. The name should include
a reference to the issue number, as well as a brief name that
suggests the big idea of the change. The issue number links the
changes you propose with the problem that they fix; and the
name provides context independent of Github.  
For instance, [PR #6](https://github.com/flatironinstitute/nomad/pull/6)
merges a feature branch named "1-return-factored-matrix". The branch
name (and pull request title) indicate that the proposed changes
fix [issue #1](https://github.com/flatironinstitute/nomad/issues/1),
a request for the algorithms to return the low-rank approximation
matrix in a factored form to save memory.
* **Develop and test your changes.** For specific advice on testing
and style, see "Developer Information" below.
* **Submit your pull request.** If you have a feature branch with
recent changes, Github will often add a banner suggesting that
you make a pull request; or you can start one directly from
[the pull request page](https://github.com/flatironinstitute/nomad/compare).
Be sure to include a good summary of your proposed changes! Writing
a good PR is a large topic, but the following key points can help:
  * *Reference the issue*. The example of PR #6 (above) begins with
  "Fixes #1" to link the corresponding Github issue.
  * *Describe the key changes* you make, and how those changes solve
  the problem. Especially for large PRs with lots of changes, it's
  important to indicate the highlights. Give your reviewers a guide
  to what is important and how best to read the changed code.
  * *Describe what you did to verify the changes*. This can (and
  should!) include adding automated unit or integration tests, but
  may also refer to any manual testing you've done, edge cases you
  considered, and so on.
  * *Provide any other details* you think reviewers should know.
* **Participate in review**. The core developers will review your
proposed changes, discuss them with you, and work with you to make
any needed improvements. Reviewing your work--especially when there
are requests for changes--can be an emotional process. We commit to
focusing on the best interests of the overall project, and ask that
you do the same. Your active participation in review and responsiveness
to requested changes is essential to keeping this process smooth and
quick.  
Once everything is in order, your changes
will be incorporated into the
official version of the package (the `trunk`) and included in the
next version update.

If your proposed changes impact the code (rather than documentation),
please note that this project makes use of specific practices and
automated tools that help ensure that the library works as expected
and that its code is readable and of high quality. These are discussed
below.


## Developer Information

This section describes design principles and practices, as well as
automated tools, used by project developers. Following them is not
required, but will greatly reduce friction for submitting changes
and make your pull request more likely to be successful.

### Environment

We make use of [black](https://black.readthedocs.io/en/stable/)
and [pylint](https://pylint.readthedocs.io/en/stable/)
for code formatting, [mypy](https://mypy.readthedocs.io/en/stable/)
for type checking, and [pytest](https://docs.pytest.org/en/7.1.x/contents.html)
for automated unit and integration tests. The tools are configured
in `pyproject.toml` so that they should require minimal command-line
flags; from the repository root, you can simply do:

```bash
black .
pylint src/
mypy .
pytest test/
```

to run the same automated checks that will be applied against incoming
pull requests.

We strive to ensure complete coverage with both
[unit](https://en.wikipedia.org/wiki/Unit_testing) and
[integration](https://en.wikipedia.org/wiki/Integration_testing) tests.


### Style

In addition to the technical elements enforced by the code
formatters above, we have the following stylistic practices:

* **Keep function bodies short.** Ideally, individual functions won't
extend beyond a screen: this helps encourage developers to break
complex processes into named, reusable parts that are easier to
understand and test.
* **Prefer functions over classes when possible.** Python's design
tends to promote object use, but
[pure functions](https://en.wikipedia.org/wiki/Pure_function) are
simpler to test and improve upon.
* **Prefer smaller files with a clearer purpose.** This makes it
easier to track changes in version control, and to group together
related functionality.
* **Prefer semantic names for variables and functions.** Notation
differs, and often a particular variable name only makes sense in
the context of a paper. Briefly-descriptive names make it easier
to reason about the parts of an algorithm or equation, and make
the code more accessible while reducing required context-switching.  
For example, the value `gamma` in Saul (2022) refers to a matrix of
means for a Gaussian model, divided by the square root
of the model variance. When referring to this value, we prefer the
semantic and mnemonic `stddev_normalized_matrix_gamma` over
plain `gamma`.
* **Prefer semantic types and type aliases when appropriate.**
Using semantic type names, especially for moderately-complex types,
makes it easier to enforce consistency and support future changes.  
For instance, many functions in the library are expected to return
vectors or matrices. These are aliased as `FloatArrayType` instead
of `numpy.typing.NDArray[numpy.float_]`. Using the alias simplifies
the import (`numpy.typing` only needs to be referenced once) and
also ensures that, should the underlying type need to change, it
can be changed once and applied everywhere.
* **Use mocking in unit tests.** Unit tests (at the function level)
should help localize problems by testing the assumptions about
fine-grained components. Mocking the functions called by a function
under test helps the test to focus on a specific expected
behavior, and ensures that it runs fast enough to be non-disruptive.
* **Provide explanatory docstrings.** While fine details of any
particular algorithm are most precisely explained in the
corresponding paper, we want a developer working
with the code to have some understanding of what a function does.
Citing a paper in a comment isn't enough to accomplish this.
In the best case, a bare citation invites a disruptive context
switch for the developer; in the worst case, the developer may
not readily understand or even have access to the paper. There's
no need to be exhaustive, but prefer some intuitive explanation in
function descriptions.

Docstrings should be present for modules, classes, and functions
(`pylint` will check for this).
We use Google style, but omit type information (since this is more
effectively documented by type annotations in the code).


## Code Organization

All package code is stored in `src/fi_nomad`.

At present, there are three main areas of functionality organized
into their own directories, in addition to the main entry point
function, `decompose`, in `entry.py`.

Providing extensive organization of atomic files can result in fairly
deep directory structure and correspondingly long `import` statements
in Python. To avoid this, the `__init__.py` files should be used to
re-export public-facing functionality.

* **kernels** defines the actual algorithms for low-rank approximation,
as well as the common interface they share.
  * `kernel_base.py` provides the Abstract Base Class that defines
  the interface all kernels are expected to follow.
  * New kernels should be added here and should implement the
  `KernelBase` interface.
  * In general, kernel classes should limit themselves to implementing
  the interface functions, calling mathematical operations defined
  elsewhere, and collecting operations that change their own data.
  Significant computation should be defined in files in the `util`
  directory, as pure functions, that will be easier to test.
* **util** defines the mathematical functions that support the kernels.
They are organized into files based on the type of operation and
the sets of kernels which might share them.
  * `base_model_free_util.py` and `gauss_model_util.py` define functions
  that are used by specific kernels.
  * `decomposition_util.py` defines utilities for matrix decomposition
  (by SVD); `initialization_util.py` defines setup functionality used
  by the main loop; `loss_util.py` and `stats_util.py` define functions
  to compute loss values and deal with probability distributions,
  respectively.
  * New kernels will likely require adding one or more `XXX_util.py`
  files for the computation functions that are unique to their
  algorithms. Of course, if it's possible for a function to be shared
  between kernels, this is ideal--avoid repeating code when possible.
* **types** defines all types for the package.
  * `enums.py` uses `Enumeration` functionality to provide consistent
  semantic labels for constant values, such as choices of kernel,
  SVD strategy, etc. New kernels will need to add to the
  `KernelStrategy` enum in this file, as well as modifying the
  factory function in `entry.py`. Essentially, where code exposes a
  choice from a number of discrete possibilities, those possibilities
  should be given a label as an enum.
  * `kernelInputTypes.py` and `kernelReturnTypes.py` define the
  data objects that are passed to and returned from kernels.
    * If a kernel needs supplemental parameters, such as algorithm-specific
  hyperparameters, they should be defined as a `NamedTuple` in
  `kernelInputTypes.py` and added to the `KernelSpecificParameters`
  type alias union.
    * Kernels that return information beyond the low-rank approximation
    matrix--such as the `variance` parameters learned by the Gaussian
    model kernels--should define a `@dataclass` inheriting from
    `KernelReturnBase` and add it to the `KernelReturnDataType`
    type alias union.
    * These type alias unions help the ABC enforce consistency
    across different kernels, ensuring a consistent interface
    across algorithms and simplifying the process of applying
    changes across the library.

The `test` directory, like the package configuration files, lives at
the same level as `src` (top level of the repository). Its internal
structure matches the structure of `src/fi_nomad`, with the exception
of the `integration_tests` directory that houses all integration tests.

