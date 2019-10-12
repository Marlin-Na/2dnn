

let data;
let labels;
let net;
let trainer;
let N;
const ITER_NUMBER_PER_PERIODIC = 2;
const PERIODIC_INTERVAL = 10;

function init_circle_data(size) {
    if (size === undefined) {
        size = 50;
    }
    data = [];
    labels = [];
    for(var i=0;i<size;i++) {
      var r = convnetjs.randf(0.0, 2.0);
      var t = convnetjs.randf(0.0, 2*Math.PI);
      data.push([r*Math.sin(t), r*Math.cos(t)]);
      labels.push(1);
    }
    for(var i=0;i<size;i++) {
      var r = convnetjs.randf(3.0, 5.0);
      //var t = convnetjs.randf(0.0, 2*Math.PI);
      var t = 2*Math.PI*i/50.0
      data.push([r*Math.sin(t), r*Math.cos(t)]);
      labels.push(0);
    }
    N = data.length;
    return;
}

function init_spiral_data(size) {
    data = [];
    labels = [];
    if (size === undefined) {
        size = 50;
    }
    for(var i=0;i<size;i++) {
        var r = i/n*5 + convnetjs.randf(-0.1, 0.1);
        var t = 1.25*i/n*2*Math.PI + convnetjs.randf(-0.1, 0.1);
        data.push([r*Math.sin(t), r*Math.cos(t)]);
        labels.push(1);
    }
    for(var i=0;i<size;i++) {
        var r = i/n*5 + convnetjs.randf(-0.1, 0.1);
        var t = 1.25*i/n*2*Math.PI + Math.PI + convnetjs.randf(-0.1, 0.1);
        data.push([r*Math.sin(t), r*Math.cos(t)]);
        labels.push(0);
    }
    N = data.length;
}


function init_net_and_trainer() {
    let layer_defs = [
        {type:'input', out_sx:1, out_sy:1, out_depth:2}, // Inputs are 2-dim data points
        {type:'fc', num_neurons:20, activation:'relu'},
        {type:'fc', num_neurons:20, activation:'relu'},
        // {type:'fc', num_neurons:6, activation: 'tanh'},
        // {type:'fc', num_neurons:2, activation: 'tanh'},
        {type:'softmax', num_classes:2},
    ];

    // create a net out of it
    net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    // Init trainer
    trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, momentum:0.1, batch_size:10, l2_decay:0.001});
}

function update() {
    let start = new Date().getTime();

    let x = new convnetjs.Vol(1,1,2);
    //x.w = data[ix];
    let avloss = 0.0;
    for (let iters=0; iters<ITER_NUMBER_PER_PERIODIC; iters++) {
        for (let ix=0; ix<N; ix++) {
            x.w = data[ix];
            let stats = trainer.train(x, labels[ix]);
            avloss += stats.loss;
        }
    }
    avloss /= N*ITER_NUMBER_PER_PERIODIC;

    let end = new Date().getTime();
    let time = end - start;

    //console.log('loss = ' + avloss + ', ' + ITER_NUMBER_PER_PERIODIC + ' cycles through data in ' + time + 'ms');
}

function plot() {
    data;
    labels;
    net;
    trainer;
    N;

    const margin = {
        left: 20,
        right: 20,
        top: 20,
        bottom: 20
    };
    const grid_dim = 30;
    let svg = d3.select("#plotsvg");
    let svg_width = svg.node().width.baseVal.value;
    let svg_height = svg.node().height.baseVal.value;
    let width = svg_width - margin.left - margin.right;
    let height = svg_height - margin.top - margin.bottom;

    let scaleX = d3.scaleLinear()
        .domain(d3.extent(data.map(d => d[0])))
        .range([margin.left, margin.left + width])
        .nice();
    let scaleY = d3.scaleLinear()
        .domain(d3.extent(data.map(d => d[1])))
        .range([margin.top + height, margin.top])
        .nice();
    let scaleColor = d3.interpolateRdYlGn;

    let grid_data = [];
    let grid_width = width/grid_dim;
    let grid_height = height/grid_dim;
    for (let i=0; i<grid_dim; i++) {
        for (let j=0; j<grid_dim; j++) {
            let canvas_x = margin.left + i * grid_width; 
            let canvas_y = margin.top + j * grid_height; 
            let x = scaleX.invert(canvas_x + grid_width/2);
            let y = scaleY.invert(canvas_y + grid_height/2);

            let dat = new convnetjs.Vol([x, y]);
            let probvol = net.forward(dat);
            let prob_class_1 = probvol.w[1];
            grid_data.push({
                canvas_x: canvas_x,
                canvas_y: canvas_y,
                x: x, y: y, prob_class_1: prob_class_1
            });
        }
    }

    let gridRectSel = svg.selectAll("rect")
        .data(grid_data);
    gridRectSel.enter()
        .append("rect")
        .merge(gridRectSel)
        .attr("width", grid_width)
        .attr("height", grid_height)
        .attr("x", d => d.canvas_x)
        .attr("y", d => d.canvas_y)
        .attr("fill", d => scaleColor(d.prob_class_1))
        .attr("stroke", d => scaleColor(d.prob_class_1));
    gridRectSel.exit().remove();
    
    let circleSel = svg.selectAll("circle")
        .data(data);
    circleSel.enter()
        .append("circle")
        .merge(circleSel)
        .attr("cx", d => scaleX(d[0]))
        .attr("cy", d => scaleY(d[1]))
        .attr("fill", (d, i) => scaleColor(labels[i]))
        .attr("stroke", "black")
        .attr("stroke-width", 2)
        .attr("r", 4);
    circleSel.exit().remove();

    let xAxis = d3.axisBottom().scale(scaleX);
    let yAxis = d3.axisLeft().scale(scaleY); 
    let xaxis_g = svg
        .selectAll("g.xaxis")
        .data([88]);
    xaxis_g
        .enter()
        .append("g")
        .attr("class", "xaxis")
        .merge(xaxis_g)
        .attr("transform", `translate(0, ${margin.top + height})`)
        .call(xAxis);
    let yaxis_g = svg
        .selectAll("g.yaxis")
        .data([43]);
    yaxis_g
        .enter()
        .append("g")
        .attr("class", "yaxis")
        .merge(yaxis_g)
        .attr("transform", `translate(${margin.left}, 0)`)
        .call(yAxis);
}

let ID = null;

function start() {

    if (ID !== null) {
        clearInterval(ID);
    }

    init_circle_data();
    init_net_and_trainer();
    plot();

    function periodic() {
        update();
        plot();
    }

    ID = setInterval(periodic, PERIODIC_INTERVAL);
}
