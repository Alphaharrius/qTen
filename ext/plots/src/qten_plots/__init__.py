"""
Plotting-oriented extension package for QTen.

`qten_plots` contains higher-level plot objects and backend adapters layered on
top of the core plotting dispatch interface provided by
[`qten.plottings`][qten.plottings]. The extension keeps plot-ready helper
objects separate from the core tensor, geometry, and symbolic packages while
registering backend-specific renderers for those core objects.

Repository usage
----------------
The main public module is [`qten_plots.plottables`][qten_plots.plottables],
which defines reusable plot objects such as
[`PointCloud`][qten_plots.plottables.PointCloud]. Backend implementation modules
provide Matplotlib and Plotly registrations for QTen's plotting dispatcher.
The generated API page links to the Plot Methods index for the registered
`obj.plot(...)` calls.

Notes
-----
Backend implementation modules are private implementation details. Users should
construct public plottable objects and call QTen's plotting interface rather
than importing backend functions directly.
"""
