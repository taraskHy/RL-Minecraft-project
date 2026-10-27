'use strict'
// prismarine-viewer's server side imports its browser renderer, which requires the native
// 'canvas' package at load time but never calls it on the server (the page renders in the
// browser). This stub satisfies that import so no native module is needed.
class Canvas {}
class Image {}
module.exports = { Canvas, Image, createCanvas: () => ({ getContext: () => ({}) }) }
