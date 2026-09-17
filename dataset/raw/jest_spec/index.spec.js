'use strict';

var chai = require('chai');
var expect = chai.expect;
// var assert = chai.assert();


var sut = require('./index');

describe('Arrow functions', function (){
  it('should return a function that returns 17.2', function(){
    var result = sut.arrows();
    expect(result).to.equal(17.2);
  });
});


describe('Object creating', function(){
  it('should create a new object from car object', function(){
    var car = { carType: 'sedan', wheels: '4'};

    car.getNumber = function(x){
      var multiplier = 4;
      if (x){
        multiplier = x;
      }
      return Math.random() * this.multiplier;
    }
    var bmw = Object.create(car);



    bmw.color = 'black';
    bmw.carType = 'saloon';
    expect(bmw).to.be.an('object');

    expect(car.carType).to.equal('sedan');

    expect(bmw.carType).to.equal('saloon');
    console.log(bmw.getNumber(20));
  });
});


describe('Inheritance', function(){
  it('should create instance of a new object', function(){
 'use strict'; 
    let animal = {
      animalType: 'mammal',
      describe () {
        return `An ${this.animalType}, with ${this.furColor} fur,
              ${this.legs} legs, and a ${this.tail} tail.`;
      }
    };

    let bird = Object.assign(Object.create(animal), {
      animalType: 'bird',
      furColor: 'grey',
      legs: 2,
      tail: 'long, bushy'
    });

    console.log(bird.describe());

  });
});
